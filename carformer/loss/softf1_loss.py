from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class SoftF1Loss(nn.Module):
    def __init__(self, avg="micro"):
        super().__init__()
        self.avg = avg

    def forward(self, y_pred, y_target):
        return softF1Loss(y_pred, y_target, avg=self.avg)


def softF1Loss(y_pred, y_target, avg="micro"):
    if len(y_target.size()) == 1:
        nlabels = y_pred.size()[-1]
        y_target = torch.eye(nlabels, device=y_target.device)[y_target]

    y_pred = y_pred.softmax(-1)

    tp = y_target * y_pred
    fp = (1 - y_target) * y_pred
    fn = y_target * (1 - y_pred)

    f1_scores = tp / (tp + 0.5 * (fp + fn) + 1e-8)

    if avg == "macro":
        f1_scores = f1_scores.sum(0) / y_target.sum(0)
    else:
        f1_scores = f1_scores.sum(1)

    return 1 - f1_scores.mean()
