#!/usr/bin/env python3

import torch
from torch.nn import functional as F

_VERY_BIG_NUMBER = 1e30


def softmax_with_mask(x: torch.Tensor, dim=None, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        x = x + (mask.float() - 1.0) * _VERY_BIG_NUMBER
    return F.softmax(x, dim)
