import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from torch.cuda.amp import autocast


def calculat_acc(output, target):
    output, target = output.view(-1, 62), target.view(-1, 62)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


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
                    albumentations.Resize(DIM, int(DIM*2.5)),
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
                    albumentations.Resize(DIM, int(DIM*2.5)),
                    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, p=1.0),
                    ToTensorV2(p=1.0),
                   ], p=1.)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate_fn(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    accs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):

            images = images.to(params['device'], non_blocking=True).float()
            target = target.to(params['device'], non_blocking=True).long()
            output = model(images)

            loss = criterion(output, target)
            acc = calculat_acc(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('Acc', acc)
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

            accs.append(acc)
    return np.mean(accs)
