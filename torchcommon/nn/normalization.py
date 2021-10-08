#!/usr/bin/env python3

"""
@author: xi
@since: 2021-08-03
"""

import torch
from torch import nn
from torch.nn import init


class LayerNorm(nn.Module):

    def __init__(self, shape, eps=1e-7):
        """Layer normalization

        :param shape: Feature shape.
        :param eps: Used to prevent divide zero.

        References:
            Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, Layer Normalization,
            https://arxiv.org/pdf/1607.06450.pdf
        """
        super(LayerNorm, self).__init__()
        self._eps = eps

        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape))

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        axes = tuple(range(1, len(x.shape)))
        mean = x.mean(axes, keepdim=True)
        var = (x ** 2).mean(axes, keepdim=True) - mean ** 2
        h = (x - mean) / (torch.sqrt(var) + self._eps)
        h = h * self.weight + self.bias
        return h


class GroupNorm(nn.Module):

    def __init__(self,
                 num_groups: int,
                 feature_size: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 device=None,
                 dtype=None):
        super(GroupNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self._num_groups = num_groups
        self._feature_size = feature_size
        self._eps = eps
        self._affine = affine

        self._group_size = feature_size // num_groups
        self._big_group_index = (self._group_size + 1) * (feature_size % num_groups)

        if self._affine:
            self.weight = nn.Parameter(torch.empty(feature_size, 1, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(feature_size, 1, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self._affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        n, c = shape[0], shape[1]
        x = x.view((n, c, -1))
        if self._big_group_index > 0:
            x1 = x[:, :self._big_group_index, :]
            x0 = x[:, self._big_group_index:, :]
            x1 = self._group_norm(x1, self._group_size + 1)
            x0 = self._group_norm(x0, self._group_size)
            x = torch.cat([x1, x0], 1)
        else:
            x = self._group_norm(x, self._group_size)
        x = self.weight * x + self.bias
        x = x.view(shape)
        return x

    def _group_norm(self, x: torch.Tensor, gs: int):
        n, d, m = x.shape
        assert d % gs == 0
        x = x.view((n, d // gs, gs, m))
        mean = x.mean((2, 3), keepdim=True)
        var = x.square().mean((2, 3), keepdim=True) - mean.square()
        x = (x - mean) / ((var + self._eps).sqrt())
        x = x.view((n, d, m))
        return x
