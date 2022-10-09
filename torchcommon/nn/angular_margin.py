#!/usr/bin/env python3

from typing import Union, Sequence

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'LargeMarginCosine',
    'AdditiveAngularMargin'
]


class LargeMarginCosine(nn.Module):

    def __init__(
            self,
            margin: Union[float, Sequence[float]] = 0.1,
            scale: float = 5.0,
            eps: float = 1e-12
    ) -> None:
        super(LargeMarginCosine, self).__init__()
        if isinstance(margin, Sequence):
            self.margin = torch.tensor(margin)
        elif isinstance(margin, float):
            self.margin = margin
        else:
            raise RuntimeError('Invalid margin type.')
        self.scale = scale
        self.eps = eps

    def forward(
            self,
            feat: torch.Tensor,  # (n, d)
            proto: torch.Tensor,  # (c, d)
            target: torch.Tensor = None  # (n, c)
    ) -> torch.Tensor:
        sim = F.normalize(feat, 2, -1, eps=self.eps) @ F.normalize(proto, 2, -1, eps=self.eps).T  # (n, c)
        if target is not None:
            if len(target.shape) == 1:
                target = F.one_hot(target, sim.shape[-1]).float()
            sim = (sim - self.margin * target)
        return sim * self.scale


class AdditiveAngularMargin(nn.Module):

    def __init__(
            self,
            margin: Union[float, Sequence[float]] = 0.1,
            scale: float = 5.0,
            eps: float = 1e-12
    ) -> None:
        super(AdditiveAngularMargin, self).__init__()
        if isinstance(margin, Sequence):
            self.margin = torch.tensor(margin)
        elif isinstance(margin, float):
            self.margin = margin
        else:
            raise RuntimeError('Invalid margin type.')
        self.scale = scale
        self.eps = eps

    def forward(
            self,
            feat: torch.Tensor,  # (n, d)
            proto: torch.Tensor,  # (c, d)
            target: torch.Tensor = None  # (n, c)
    ) -> torch.Tensor:
        sim = F.normalize(feat, 2, -1, eps=self.eps) @ F.normalize(proto, 2, -1, eps=self.eps).T  # (n, c)
        if target is not None:
            if len(target.shape) == 1:
                target = F.one_hot(target, sim.shape[-1]).float()
            theta = torch.arccos(sim)
            theta = torch.clip(theta + target * self.margin, 0, torch.pi)
            sim = torch.cos(theta)
        return sim * self.scale
