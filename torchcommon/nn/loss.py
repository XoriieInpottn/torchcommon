#!/usr/bin/env python3

from torch import nn
from torch.nn import functional as F


class MinimalEntropyLoss(nn.Module):

    def __init__(self, original_loss, lmd=1.0, eps=1e-30):
        super(MinimalEntropyLoss, self).__init__()
        self.original_loss = original_loss
        self.lmd = lmd
        self.eps = eps

    def forward(self, output, target):
        loss = self.original_loss(output, target)
        if self.lmd is not None and self.lmd != 0:
            p = F.softmax(output, -1)
            h = -(p * (p + self.eps).log()).sum(-1).mean()
            loss = loss + self.lmd * h
        return loss
