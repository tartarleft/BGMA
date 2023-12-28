#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.common_types import _size_2_t

from .linear import TimeDelay
from .functional import statistics_pooling


class LinearLayer(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            inplace: bool = False,
            seq: Tuple[str] = ('linear', 'relu', 'bn')
        ) -> None:
        super(LinearLayer, self).__init__()
        features = in_features
        for name in seq:
            if name == 'linear':
                self.linear = nn.Linear(in_features, out_features, bias=bias)
                features = out_features
            elif name == 'bn':
                self.bn = nn.BatchNorm1d(features)
        self.inplace = inplace
        self.seq = seq

    def forward(self, x: Tensor) -> Tensor:
        for name in self.seq:
            if name == 'linear':
                x = self.linear(x)
            elif name == 'relu':
                x = F.relu(x, inplace=self.inplace)
            elif name == 'bn':
                x = self.bn(x)
        return x


class TdnnLayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: _size_2_t = 0,
            dilation: int = 1,
            bias: bool = True,
            impl: str = 'conv',
            inplace: bool = False,
            seq: Tuple[str] = ('tdnn', 'relu', 'bn')
        ) -> None:
        super(TdnnLayer, self).__init__()
        channels = in_channels
        for name in seq:
            if name == 'tdnn':
                self.tdnn = TimeDelay(
                    channels, out_channels, kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias)
                channels = out_channels
            elif name == 'bn':
                self.bn = nn.BatchNorm1d(channels)
        self.inplace = inplace
        self.seq = seq

    def forward(self, x: Tensor) -> Tensor:
        for name in self.seq:
            if name == 'tdnn':
                x = self.tdnn(x)
            elif name == 'relu':
                x = F.relu(x, inplace=self.inplace)
            elif name == 'bn':
                x = self.bn(x)
        return x


class Tdnn2Layer(nn.Module):

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: _size_2_t = 0,
            dilation: int = 1,
            bias: bool = True,
            impl: str = 'conv',
            inplace: bool = False,
            scale: int = 4,
            seq: Tuple[str] = ('tdnn', 'relu', 'bn')
        ) -> None:
        super(Tdnn2Layer, self).__init__()
        if channels % scale != 0:
            raise ValueError('channels must be divisible by scale')
        self.scale = scale
        self.width = channels // scale
        self.num_layers = max(1, scale - 1)
        self.functions = nn.ModuleList([
            TdnnLayer(
                self.width, self.width, kernel_size, stride, padding,
                dilation, bias, impl, inplace, seq
            ) for i in range(self.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        spx = torch.split(x, self.width, dim=1)
        res = []
        for i in range(self.num_layers):
            if i == 0:
                x = spx[i]
            else:
                x = spx[i] + x
            x = self.functions[i](x)
            res.append(x)
        if self.scale != 1:
            res.append(spx[self.num_layers])
        return torch.cat(res, dim=1)


class SqueezeExcite(nn.Module):

    def __init__(
            self,
            channels: int,
            reduction: int = 4
        ):
        super(SqueezeExcite, self).__init__()
        if channels % reduction != 0:
            raise ValueError(
                'channels must be divisible by reduction')
        self.reduction = reduction
        self.linear1 = nn.Linear(channels, channels // reduction)
        self.linear2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: Tensor) -> Tensor:
        dims = list(range(x.dim()))[2:]
        out = F.relu(self.linear1(torch.mean(x, dim=dims)))
        weight = torch.sigmoid(self.linear2(out))
        for dim in dims:
            weight = weight.unsqueeze(dim=dim)
        return x * weight



class MeanConvT1d(nn.Module):
    def __init__(self):
        super(GaussianConvT1d, self).__init__()
        kernel = torch.FloatTensor([4/7,7/7,4/7]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        # (B, C, W)
        x = x.unsqueeze(1)
        x = F.conv_transpose1d(x, self.weight, padding=1) / 3
        x = x.squeeze(1)
        return x

class ContextAwareMask(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            context_size: int,
            bias: bool = True,
            memory_efficient=False,
            seq: Tuple[str] = ('tdnn', 'relu', 'bn')
        ) -> None:
        super(ContextAwareMask, self).__init__()
        if seq[0] == 'bn':
            self.init_bn = nn.BatchNorm1d(in_channels)
        self.linear = nn.Linear(in_channels * 2, context_size, bias=True)
        self.tdnn1 = TimeDelay(in_channels, context_size, 1, bias=False)
        self.bn = nn.BatchNorm1d(context_size)
        self.tdnn2 = TimeDelay(context_size, out_channels, 1, bias=True)
        self.memory_efficient = memory_efficient
        self.seq = seq

    def bn_function(self, x: Tensor) -> Tensor:
        if self.seq[0] == 'bn':
            x = F.relu(self.init_bn(x), inplace=self.memory_efficient)
        context = self.linear(statistics_pooling(x, 2))
        x = self.tdnn1(x) + torch.unsqueeze(context, 2)
        if self.seq[0] == 'bn':
            return self.bn(F.relu(x))
        return F.relu(self.bn(x))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.memory_efficient:
            out = cp.checkpoint(self.bn_function, x)
        else:
            out = self.bn_function(x)
        return torch.sigmoid(self.tdnn2(out))
