#!/usr/bin/env python3

"""
@author: xi
@since: 2021-12-16
"""

from typing import Union, Tuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'ConvBlock2d',
    'Focus2d',
    'Bottleneck2d',
    'CSP2d',
    'SPP2d',
    'FastSPP2d',
    'PACFusion2d',
    'PACPyramidNetwork2d',
    'ConvHead2d',
    'PFEDecoder'
]


def ensure_tuple(t):
    return (t, t) if isinstance(t, int) else t


def fuse_conv_and_bn(conv, bn):
    # https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
    # init
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.matmul(w_bn, b_conv) + b_bn)
    # we're done
    return fusedconv


class PSwish(nn.SiLU):

    def __init__(self, init=0.0):
        super(PSwish, self).__init__()
        self.weight = nn.Parameter(torch.empty(()).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.weight.sigmoid() + 0.5
        return super(PSwish, self).forward(a * x) / a


class ConvBlock2d(nn.Module):
    """Standard convolution"""

    DefaultNorm = nn.BatchNorm2d
    DefaultNonLin = nn.SiLU

    def __init__(
            self,
            ch_in: int,
            ch_out: int,
            kernel: Union[int, Tuple[int]] = 1,
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = None,
            groups: int = 1,
            Norm=None,
            NonLin=None
    ) -> None:
        super(ConvBlock2d, self).__init__()
        kernel = ensure_tuple(kernel)
        stride = ensure_tuple(stride)
        if padding is None:
            padding = tuple(_k // 2 for _k in kernel)
        padding = ensure_tuple(padding)

        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        if Norm is None:
            Norm = ConvBlock2d.DefaultNorm
        self.norm = Norm(ch_out)

        if NonLin is None:
            NonLin = ConvBlock2d.DefaultNonLin
        self.non_lin = NonLin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.non_lin(self.norm(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.non_lin(self.conv(x))


class Focus2d(ConvBlock2d):
    """Focus wh information into c-space"""

    def __init__(
            self,
            ch_in,
            ch_out,
            kernel=1,
            stride=1,
            padding=None,
            groups=1,
            Norm=None,
            NonLin=None
    ) -> None:
        super(Focus2d, self).__init__(
            ch_in=ch_in * 4,
            ch_out=ch_out,
            kernel=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            Norm=Norm,
            NonLin=NonLin,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = x[..., ::2, ::2]  # (n, c, h/2, w/2)
        h2 = x[..., 1::2, ::2]  # (n, c, h/2, w/2)
        h3 = x[..., ::2, 1::2]  # (n, c, h/2, w/2)
        h4 = x[..., 1::2, 1::2]  # (n, c, h/2, w/2)
        h = torch.cat([h1, h2, h3, h4], 1)  # (n, 4c, h/2, w/2)
        return super(Focus2d, self).forward(h)


class Bottleneck2d(nn.Sequential):
    """Standard bottleneck"""

    def __init__(
            self,
            ch_in: int,
            ch_out: int,
            shortcut: bool = True,
            kernels: Tuple[int, int] = (1, 3),
            groups: int = 1,
            expansion: float = 0.5,
            dropout=0.0,
            Norm=None,
            NonLin=None
    ) -> None:
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.shortcut = shortcut and (ch_out == ch_in)
        self.kernels = kernels
        self.groups = groups
        self.expansion = expansion
        self.dropout = dropout
        self.Norm = Norm
        self.NonLin = NonLin

        ch_h = int(ch_out * expansion)

        super(Bottleneck2d, self).__init__(
            ConvBlock2d(self.ch_in, ch_h, self.kernels[0], Norm=self.Norm, NonLin=self.NonLin),
            nn.Dropout(self.dropout) if self.dropout else nn.Identity(),
            ConvBlock2d(ch_h, self.ch_out, self.kernels[1], groups=self.groups, Norm=self.Norm, NonLin=self.NonLin)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super(Bottleneck2d, self).forward(x)
        return y + x if self.shortcut else y


class CSP2d(nn.Module):
    """CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            ch_in: int,
            ch_out: int,
            num_bottlenecks: int = 1,
            shortcut: bool = True,
            groups: int = 1,
            expansion: float = 0.5,
            Norm=None,
            NonLin=None
    ) -> None:
        super(CSP2d, self).__init__()
        ch_h = int(ch_out * expansion)
        self.conv0 = ConvBlock2d(ch_in, ch_h, Norm=Norm, NonLin=NonLin)
        self.conv1 = ConvBlock2d(ch_in, ch_h, Norm=Norm, NonLin=NonLin)
        self.conv2 = ConvBlock2d(ch_h * 2, ch_out, Norm=Norm, NonLin=NonLin)
        self.bottlenecks = nn.Sequential(*(
            Bottleneck2d(ch_h, ch_h, shortcut=shortcut, groups=groups, expansion=1.0, Norm=Norm, NonLin=NonLin)
            for _ in range(num_bottlenecks)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.bottlenecks(self.conv0(x))
        h1 = self.conv1(x)
        h = torch.cat([h0, h1], 1)
        return self.conv2(h)


class SPP2d(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer
    https://arxiv.org/abs/1406.4729
    """

    def __init__(
            self,
            ch_in: int,
            ch_out: int,
            kernels: Union[Tuple[int]] = (5, 9, 13),
            Norm=None,
            NonLin=None
    ) -> None:
        super(SPP2d, self).__init__()
        ch_h = ch_in // 2
        self.conv0 = ConvBlock2d(ch_in, ch_h, Norm=Norm, NonLin=NonLin)
        self.conv1 = ConvBlock2d(ch_h * (len(kernels) + 1), ch_out, Norm=Norm, NonLin=NonLin)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=(k, k), stride=1, padding=(k // 2, k // 2))
            for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv0(x)
        h = torch.cat([h, *(pool(h) for pool in self.pools)], 1)
        return self.conv1(h)


class FastSPP2d(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5
    by Glenn Jocher
    Equivalent to SPP(k=(5, 9, 13))
    """

    def __init__(
            self,
            ch_in: int,
            ch_out: int,
            kernel: int = 5,
            Norm=None,
            NonLin=None
    ) -> None:
        super(FastSPP2d, self).__init__()
        ch_h = ch_in // 2
        self.conv0 = ConvBlock2d(ch_in, ch_h, Norm=Norm, NonLin=NonLin)
        self.conv1 = ConvBlock2d(ch_h * 4, ch_out, Norm=Norm, NonLin=NonLin)
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        h0 = self.pool(x)
        h1 = self.pool(h0)
        h2 = self.pool(h1)
        h = torch.cat([x, h0, h1, h2], 1)
        return self.conv1(h)


class PACFusion2d(nn.Module):
    """Project, weighted Add and Conv (PAC) Fusion Module.
    """

    def __init__(
            self,
            ch_in_list: Sequence[int],
            ch_hid: int,
            ch_out: int,
            hw_out: Union[int, Tuple[int, int], None] = None,
            depth: int = 1,
            weighted: bool = True,
            shortcut: bool = True,
            inter_mode='nearest',
            dropout=0.0,
            Norm=None,
            NonLin=None
    ) -> None:
        super(PACFusion2d, self).__init__()
        self.ch_in_list = list(ch_in_list)
        self.ch_hid = ch_hid
        self.ch_out = ch_out
        self.hw_out = (hw_out, hw_out) if isinstance(hw_out, int) else hw_out
        self.depth = depth
        self.weighted = weighted
        self.shortcut = shortcut
        self.inter_mode = inter_mode
        self.dropout = dropout
        self.Norm = Norm
        self.NonLin = NonLin

        assert self.inter_mode in {'nearest', 'bilinear'}

        # projection
        self.projects = nn.ModuleList()
        for ch_in in self.ch_in_list:
            if ch_in != self.ch_hid:
                self.projects.append(ConvBlock2d(
                    ch_in,
                    self.ch_hid,
                    Norm=self.Norm,
                    NonLin=self.NonLin
                ))
            else:
                self.projects.append(nn.Identity())

        # weighting and merge
        if len(self.ch_in_list) > 1:
            self.weights = nn.Parameter(torch.zeros(
                (len(self.ch_in_list), self.ch_hid),
                dtype=torch.float32)
            ) if self.weighted else None
            self.merge = nn.Sequential(*(
                Bottleneck2d(
                    self.ch_hid,
                    self.ch_hid,
                    expansion=0.5,
                    shortcut=self.shortcut,
                    dropout=self.dropout,
                    Norm=self.Norm,
                    NonLin=self.NonLin
                ) for _ in range(self.depth)
            ))
        else:
            self.weights = None
            self.merge = None

        # output
        if self.ch_hid != self.ch_out:
            self.out = ConvBlock2d(
                self.ch_hid,
                self.ch_out,
                Norm=self.Norm,
                NonLin=self.NonLin
            )
        else:
            self.out = None

    def forward(self, x_list: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(x_list) == len(self.projects)
        hw_out = self.hw_out if self.hw_out is not None else tuple(x_list[0].shape[2:4])
        y_list, shortcut_list = [], []
        for x, proj in zip(x_list, self.projects):
            _, c, h, w = x.shape
            if h == hw_out[0] and w == hw_out[1]:
                if c == self.ch_out:
                    shortcut_list.append(x)
                y_list.append(proj(x))
            else:
                y_list.append(self._interpolate(proj(x), hw_out))

        if self.weights is not None:
            y = torch.einsum('m n d h w, m d -> n d h w', torch.stack(y_list), 1.0 - self.weights)
        else:
            y = sum(y_list) if len(y_list) > 1 else y_list[0]

        if self.merge is not None:
            y = self.merge(y)

        if self.out is not None:
            y = self.out(y)

        if len(shortcut_list) != 0 and self.shortcut:
            y = y + sum(shortcut_list)
        return y

    def _interpolate(self, x: torch.Tensor, hw: Tuple[int, int]):
        if self.inter_mode == 'nearest':
            return F.interpolate(x, hw, mode='nearest')
        elif self.inter_mode == 'bilinear':
            return F.interpolate(x, hw, mode='bilinear', align_corners=True)
        else:
            raise RuntimeError(f'Invalid interpolate mode {self.inter_mode}.')


class PACPyramidNetwork2d(nn.Module):

    def __init__(
            self,
            ch_in_list: Sequence[int],
            ch_hid_list: Sequence[int],
            ch_out_list: Sequence[int],
            ch_glob: int = None,
            hw_out_list: Union[Sequence[Union[int, Tuple[int, int]]], None] = None,
            depth: int = 1,
            inter_mode='nearest',
            dropout=0.0,
            Norm=None,
            NonLin=None
    ) -> None:
        super(PACPyramidNetwork2d, self).__init__()
        self.ch_in_list = list(ch_in_list)
        self.ch_hid_list = list(ch_hid_list)
        self.ch_out_list = list(ch_out_list)
        self.ch_glob = ch_glob
        self.hw_out_list = list(hw_out_list) if hw_out_list is not None else None
        self.inter_mode = inter_mode
        self.depth = depth
        self.dropout = dropout
        self.Norm = Norm
        self.NonLin = NonLin

        assert len(self.ch_in_list) == len(self.ch_out_list)
        assert self.inter_mode in {'nearest', 'bilinear'}

        self.fusions = nn.ModuleList()
        ch_last = None
        for i, (ch_in, ch_hid, ch_out) in enumerate(zip(self.ch_in_list, self.ch_hid_list, self.ch_out_list)):
            self.fusions.append(PACFusion2d(
                ch_in_list=PACPyramidNetwork2d._make_inputs(ch_in, ch_last, self.ch_glob),
                ch_hid=ch_hid,
                ch_out=ch_out,
                hw_out=self.hw_out_list[i] if self.hw_out_list is not None else None,
                depth=self.depth,
                inter_mode=self.inter_mode,
                dropout=self.dropout,
                Norm=self.Norm,
                NonLin=self.NonLin
            ))
            ch_last = ch_out

    def forward(
            self,
            x_list: Sequence[torch.Tensor],
            g: torch.Tensor = None
    ) -> Sequence[torch.Tensor]:
        y_list = []
        y = None
        for x, fusion in zip(x_list, self.fusions):
            y = fusion(PACPyramidNetwork2d._make_inputs(x, y, g))
            y_list.append(y)
        return y_list

    @staticmethod
    def _make_inputs(x_in, x_hid, x_glob) -> Sequence:
        if x_glob is None:
            if x_hid is None:
                return [x_in]
            else:
                return [x_in, x_hid]
        else:
            if x_hid is None:
                return [x_in, x_glob]
            else:
                return [x_in, x_hid, x_glob]


class ConvHead2d(nn.Sequential):

    def __init__(
            self,
            ch_in: int,
            ch_hid: int,
            ch_out: int,
            kernel_size: Union[Tuple[int], int] = 3,
            dropout: float = 0.1,
            Norm=None,
            NonLin=None
    ) -> None:
        self.ch_in = ch_in
        self.ch_hid = ch_hid
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.Norm = Norm
        self.NonLin = NonLin
        super(ConvHead2d, self).__init__(
            ConvBlock2d(self.ch_in, self.ch_hid, self.kernel_size, Norm=self.Norm, NonLin=self.NonLin),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(self.ch_hid, self.ch_out, (1, 1), (1, 1), (0, 0), bias=True)
        )


class PFEDecoder(nn.Module):
    """Prior guided Feature Enrichment (PFE) Decoder.
    """

    def __init__(
            self,
            ch_in_list: Sequence[int],
            ch_hid: int,
            ch_out: int,
            ch_prior: Union[int, None],
            hw_hid_list: Union[Sequence[Union[int, Tuple[int, int]]], None],
            hw_out: Union[int, Tuple[int, int], None],
            depth: int = 4,
            inter_mode='bilinear',
            dropout: float = 0.1,
            Norm=None,
            NonLin=None
    ) -> None:
        """Prior guided Feature Enrichment (PFE) Decoder.

        Args:
            ch_in_list: Number of channels of inputs from different scales.
            ch_hid: Number of channels of hidden feature maps.
            ch_out: Number of channels of outputs, e.g., num_classes.
            ch_prior: Number of channels of prior feature map.
            hw_hid_list: Sizes (height and width) of the scales.
            hw_out: Final output size (height and width).
            depth: Number of bottleneck modules in a fusion operation.
            inter_mode: Interpolate mode. 'bilinear' usually gives better performance, while 'nearest' is faster.
            dropout: Probability to dropout.
            Norm: Normalization module. Default is nn.BatchNorm2d.
            NonLin: Activation function. Default is nn.SiLU.
        """
        super(PFEDecoder, self).__init__()
        self.ch_in_list = list(ch_in_list)
        self.ch_hid = ch_hid
        self.ch_out = ch_out
        self.ch_prior = ch_prior
        self.hw_hid_list = list(hw_hid_list) if hw_hid_list is not None else None
        self.hw_out = hw_out
        self.depth = depth
        self.inter_mode = inter_mode
        self.dropout = dropout
        self.Norm = Norm
        self.NonLin = NonLin

        num_ins = len(self.ch_in_list)
        assert self.hw_hid_list is None or len(self.hw_hid_list) == num_ins
        self.fpn = PACPyramidNetwork2d(
            ch_in_list=ch_in_list,
            ch_hid_list=[self.ch_hid] * num_ins,
            ch_out_list=[self.ch_hid] * num_ins,
            ch_glob=self.ch_prior,
            hw_out_list=self.hw_hid_list,
            depth=self.depth,
            inter_mode=self.inter_mode,
            dropout=self.dropout,
            Norm=self.Norm,
            NonLin=self.NonLin
        )
        self.fusion = PACFusion2d(
            ch_in_list=[self.ch_hid] * num_ins,
            ch_hid=self.ch_hid,
            ch_out=self.ch_hid,
            hw_out=self.hw_out,
            depth=self.depth,
            inter_mode=self.inter_mode,
            dropout=self.dropout,
            Norm=self.Norm,
            NonLin=self.NonLin
        )

        self.fpn_heads = nn.ModuleList()
        for _ in range(num_ins):
            self.fpn_heads.append(ConvHead2d(
                ch_in=self.ch_hid,
                ch_hid=self.ch_hid,
                ch_out=self.ch_out,
                dropout=self.dropout,
                Norm=self.Norm,
                NonLin=self.NonLin
            ))
        self.fusion_head = ConvHead2d(
            ch_in=self.ch_hid,
            ch_hid=self.ch_hid,
            ch_out=self.ch_out,
            dropout=self.dropout,
            Norm=self.Norm,
            NonLin=self.NonLin
        )

    def forward(
            self,
            x_list: Sequence[torch.Tensor],
            prior: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        assert len(x_list) == len(self.ch_in_list)

        y_list = self.fpn(x_list, prior)
        if self.hw_out is None:
            # if hw_out is not specified, ensure the biggest size is used
            y_list.sort(key=lambda _a: -_a.shape[2])
        y = self.fusion(y_list)
        y = self.fusion_head(y)
        if self.training:
            y_list = list(map(
                lambda _a: _a[1](_a[0]),
                zip(y_list, self.fpn_heads)
            ))
        else:
            y_list = []
        return y, y_list
