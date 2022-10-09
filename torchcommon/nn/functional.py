#!/usr/bin/env python3

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

_VERY_BIG_NUMBER = 1e30


def softmax_with_mask(x: torch.Tensor, dim=None, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        x = x + (mask.float() - 1.0) * _VERY_BIG_NUMBER
    return F.softmax(x, dim)


def make_adv(
        model: nn.Module,
        x: torch.Tensor,
        xi: float = 10.0,
        eps: float = 2.0,
        ip: int = 1
) -> torch.Tensor:
    with torch.no_grad():
        h = model(x)
        if not isinstance(h, torch.Tensor):
            h = h[-1]
        h = h.detach()

    d = F.normalize(torch.rand(x.shape, device=x.device).sub(0.5), 2, 1)

    mode = model.training
    model.train(False)
    for _ in range(ip):
        d.requires_grad_()
        h_hat = model(x + xi * d)
        if not isinstance(h_hat, torch.Tensor):
            h_hat = h_hat[-1]
        adv_distance = -(F.normalize(h_hat, 2, 1) * F.normalize(h, 2, 1)).sum(1).mean()
        adv_distance.backward()
        d = F.normalize(d + d.grad, 2, 1)
        model.zero_grad()
    model.train(mode)
    return x + eps * d.detach()


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d
