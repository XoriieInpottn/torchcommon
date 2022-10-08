#!/usr/bin/env python3

"""
@author: xi
@since: 2021-08-03
"""

import torch
from torch import nn

__all__ = [
    'AdaptiveGroupNorm'
]


class AdaptiveGroupNorm(nn.Module):

    def __init__(
            self,
            num_groups: int,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super(AdaptiveGroupNorm, self).__init__()
        small_group_size = num_channels // num_groups
        big_group_size = small_group_size + 1

        self.num_groups_big = num_channels % num_groups
        self.num_channels_big = big_group_size * self.num_groups_big
        self.norm_big = nn.GroupNorm(
            self.num_groups_big,
            self.num_channels_big,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype
        ) if self.num_groups_big != 0 else None

        self.num_groups = num_groups - self.num_groups_big
        self.num_channels = num_channels - self.num_channels_big
        self.norm = nn.GroupNorm(
            self.num_groups,
            self.num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor):
        if self.norm_big is None:
            return self.norm(x)
        else:
            x1 = x[:, :self.num_channels, ...]
            x2 = x[:, self.num_channels:, ...]
            return torch.cat([
                self.norm(x1),
                self.norm_big(x2)
            ], 1)
