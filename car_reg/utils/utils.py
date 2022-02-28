import torch
from collections import defaultdict
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def acc_score(output, target):
    y_pred = torch.argmax(output, 1).detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    return (y_pred == target).mean()


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


# albumentations
def get_train_transforms(params):
    DIM = params['im_size']
    return albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=DIM, p=1.0),
                    albumentations.Resize(DIM, DIM),
                    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
                    albumentations.OneOf([
                        albumentations.OpticalDistortion(p=0.5),
                        albumentations.GridDistortion(p=0.5),
                    ], p=0.5),
                    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, p=1.0),
                    albumentations.Cutout(num_holes=4, max_h_size=40, max_w_size=100, p=0.5),
                    ToTensorV2(p=1.0),
                                ], p=1.)


def get_valid_transforms(params):
    DIM = params['im_size']
    return albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=DIM, p=1.0),
                    albumentations.Resize(DIM, DIM),
                    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, p=1.0),
                    ToTensorV2(p=1.0),
                   ], p=1.)


def train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler=None):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)

    for i, (images, target) in enumerate(stream, start=1):

        images = images.to(params['device'], non_blocking=True).float()
        target = target.to(params['device'], non_blocking=True).long()

        output = model(images)
        loss = criterion(output, target)

        acc = acc_score(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('Acc', acc)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")


def validate_fn(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):

            images = images.to(params['device'], non_blocking=True).float()
            target = target.to(params['device'], non_blocking=True).long()
            output = model(images)

            loss = criterion(output, target)
            acc = acc_score(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('Acc', acc)
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

            targets = (target.detach().cpu().numpy()).tolist()
            outputs = (torch.argmax(output, 1).detach().cpu().numpy()).tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)
    return final_outputs, final_targets


def train_fn_amp(train_loader, model, criterion, optimizer, epoch, params, scaler, scheduler=None):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)

    for i, (images, target) in enumerate(stream, start=1):

        images = images.to(params['device'], non_blocking=True).float()
        target = target.to(params['device'], non_blocking=True).long()

        with autocast():

            output = model(images)
            loss = criterion(output, target)

            acc = acc_score(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('Acc', acc)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

        if scheduler is not None:
            scheduler.step()
