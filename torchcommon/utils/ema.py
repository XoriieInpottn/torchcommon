#!/usr/bin/env python3


import math
from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model: nn.Module, decay=0.9999, num_updates=0):
        self.model = deepcopy(model).eval()  # FP32 EMA
        for p in self.model.parameters():
            p.requires_grad = False
        self.num_updates = num_updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

    def update(self, model):
        with torch.no_grad():
            self.num_updates += 1
            decay = self.decay(self.num_updates)

            state_dict = model.state_dict()
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1 - decay) * state_dict[k].detach()
