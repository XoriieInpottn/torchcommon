#!/usr/bin/env python3

"""
@author: xi
@since: 2022-04-11
"""

import math
from typing import Sequence

import torch
from torch import nn


class ResnetAdapter(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super(ResnetAdapter, self).__init__()
        layer0_list = [model.conv1, model.bn1]
        if hasattr(model, 'relu'):
            layer0_list.append(getattr(model, 'relu'))
        if hasattr(model, 'relu1'):
            layer0_list.append(getattr(model, 'relu1'))
        if hasattr(model, 'maxpool'):
            layer0_list.append(getattr(model, 'maxpool'))

        self.layers = nn.ModuleList([
            nn.Sequential(*layer0_list),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        ])

        # the following code will inference the net to get feature size
        # setting the backbone to eval mode to prevent its BN layers from being corrupted
        self.ch_out_list = []
        self.stride_list = []
        state = self.training
        self.train(False)
        for output in self(torch.rand((1, 3, 512, 512), dtype=torch.float32)):
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4
            _, c, h, w = output.shape
            assert h == w
            self.ch_out_list.append(int(c))
            self.stride_list.append(math.ceil(512 / h))
        self.train(state)

    def forward(self, x: torch.Tensor, depth: int = None) -> Sequence[torch.Tensor]:
        if depth is None:
            depth = len(self.layers)
        ys = []
        y = x
        for i in range(depth):
            y = self.layers[i](y)
            ys.append(y)
        return ys
