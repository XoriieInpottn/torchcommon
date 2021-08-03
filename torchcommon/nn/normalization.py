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
