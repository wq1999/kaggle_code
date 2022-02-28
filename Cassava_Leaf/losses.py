from functools import partial
from torch.nn.modules.loss import _Loss
import math
from typing import Optional
import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma=2.0,
    alpha: Optional[float] = 0.25,
    reduction="mean",
    normalized=False,
    reduced_threshold: Optional[float] = None,
) -> torch.Tensor:
    target = target.type(input.type())
    #target = target.squeeze(1)

    logpt = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if normalized:
        norm_factor = focal_term.sum() + 1e-5
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class FocalLoss(_Loss):
    def __init__(
        self, alpha=None, gamma=2, ignore_index=None, reduction="mean", normalized=False, reduced_threshold=None
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.focal_loss_fn(cls_label_input, cls_label_target)
        return loss
