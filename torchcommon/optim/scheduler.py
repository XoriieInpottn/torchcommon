#!/usr/bin/env python3

"""
@author: xi
@since: 2022-03-22
"""

import math

__all__ = [
    'CosineWarmupDecay',
    'LRScheduler',
    'MomentumScheduler'
]

from torch.optim import Optimizer


class CosineWarmupDecay(object):

    def __init__(
            self,
            num_steps,
            max_value=1.0,
            min_value=0.0,
            warm_up_proportion=0.01,
            pow_warm_up=None,
            pow_annealing=2.0
    ) -> None:
        self._num_steps = num_steps
        self._max_value = max_value
        self._min_value = min_value
        self._warm_up_proportion = warm_up_proportion
        self._pow_warm_up = pow_warm_up
        self._pow_annealing = pow_annealing
        self._warm_up_steps = int(self._warm_up_proportion * self._num_steps)

    def __len__(self):
        return self._num_steps

    def __getitem__(self, i: int) -> float:
        if not (0 <= i < self._num_steps):
            raise IndexError()

        if i < self._warm_up_steps:
            i = i - self._warm_up_steps + 1
            value = (math.cos(i * math.pi / self._warm_up_steps) + 1.0) * 0.5
            if self._pow_warm_up is not None and self._pow_warm_up != 1.0:
                value = math.pow(value, self._pow_warm_up)
        else:
            i = i - self._warm_up_steps
            value = (math.cos(i * math.pi / (self._num_steps - self._warm_up_steps)) + 1.0) * 0.5
            if self._pow_annealing is not None and self._pow_annealing != 1.0:
                value = math.pow(value, self._pow_annealing)

        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class ScaledScheduler(object):

    def __init__(self, optimizer: Optimizer, factor_list, init_step=0):
        self.optimizer = optimizer
        self.factor_list = factor_list
        self.init_step = init_step

        self._i = init_step

        self.step()

    def reset(self):
        self._i = self.init_step

    def step(self, i=None):
        if i is None:
            i = self._i
            if self._i < len(self.factor_list) - 1:
                self._i += 1
        else:
            i = max(i, 0)
            i = min(i, len(self.factor_list) - 1)

        self.apply(self.factor_list[i])

    def apply(self, factor):
        raise NotImplementedError()


class LRScheduler(ScaledScheduler):

    def apply(self, factor):
        for group in self.optimizer.param_groups:
            if 'lr' not in group:
                continue
            if 'init_lr' not in group:
                group['init_lr'] = group['lr']
            group['lr'] = group['init_lr'] * factor


class MomentumScheduler(ScaledScheduler):

    def apply(self, factor):
        for group in self.optimizer.param_groups:
            if 'momentum' in group:
                if 'init_momentum' not in group:
                    group['init_momentum'] = group['momentum']
                group['momentum'] = group['init_momentum'] * factor
            elif 'betas' in group:
                if 'init_betas' not in group:
                    group['init_betas'] = group['betas']
                init_betas = group['init_betas']
                group['betas'] = (init_betas[0] * factor, init_betas[1])
