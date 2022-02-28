import torch
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def divice_norm_bias(model):
    norm_bias_params = []
    non_norm_bias_params = []
    except_wd_layers = ['norm', '.bias']
    for n, p in model.model.named_parameters():
        if any([nd in n for nd in except_wd_layers]):
            norm_bias_params.append(p)
        else:
            non_norm_bias_params.append(p)
    return norm_bias_params, non_norm_bias_params


def usr_rmse_score(output, target):
    y_pred = torch.sigmoid(output).cpu()
    y_pred = y_pred.detach().numpy() * 100
    target = target.cpu() * 100

    return mean_squared_error(target, y_pred, squared=False)


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def get_scheduler(optimizer, scheduler_params, train_df):
    if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params['T_0'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )
    elif scheduler_params['scheduler_name'] == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scheduler_params['max_lr'],
            pct_start=0.25,
            steps_per_epoch=int(((scheduler_params['num_fold']-1) * train_df.shape[0]) / (scheduler_params['num_fold'] * scheduler_params['batch_size'])) + 1,
            epochs=scheduler_params['epochs'],
        )

    elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params['T_max'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )
    return scheduler


# albumentations
def get_train_transforms(params):
    dim = params['im_size']
    return albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=dim, p=1.0),
            albumentations.RandomCrop(height=dim, width=dim, p=1.0),
            # albumentations.VerticalFlip(p=0.5),
            # albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        ]
    )


def get_valid_transforms(params):
    dim = params['im_size']
    return albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=dim, p=1.0),
        albumentations.CenterCrop(height=dim, width=dim, p=1.0),
    ], p=1.0)


def train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler=None):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)

    for i, (images, target) in enumerate(stream, start=1):

        images = images.to(params['device'], non_blocking=True)
        target = target.to(params['device'], non_blocking=True).float().view(-1, 1)

        output = model(images)
        loss = criterion(output, target)

        rmse_score = usr_rmse_score(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('RMSE', rmse_score)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")


def validate_fn(val_loader, model, criterion, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):

            images = images.to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            rmse_score = usr_rmse_score(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('RMSE', rmse_score)
            stream.set_description(f"Valid. {metric_monitor}")

            targets = (target.detach().cpu().numpy() * 100).tolist()
            outputs = (torch.sigmoid(output).detach().cpu().numpy() * 100).tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)
    return final_outputs, final_targets


def train_fn_amp(train_loader, model, criterion, optimizer, epoch, params, scaler, scheduler=None):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)

    for i, (images, dense, target) in enumerate(stream, start=1):

        images = images.to(params['device'], non_blocking=True)
        dense = dense.to(params['device'], non_blocking=True)
        target = target.to(params['device'], non_blocking=True).float().view(-1, 1)

        with autocast():

            output = model(images, dense)
            loss = criterion(output, target)

            rmse_score = usr_rmse_score(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('RMSE', rmse_score)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")
