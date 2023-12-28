#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn, Tensor
from torch.nn.common_types import _size_2_t

from ..components import TdnnLayer
from ..linear import TimeDelay
from ..pooling import StatsPool


class DenseTdnnLayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bn_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: _size_2_t = 0,
            dilation: int = 1,
            bias: bool = True,
            memory_efficient: bool = False,
            seq: Tuple[str] = ('tdnn', 'relu', 'bn')
        ) -> None:
        super(DenseTdnnLayer, self).__init__()
        self.memory_efficient = memory_efficient
        self.bn_function = TdnnLayer(
            in_channels, bn_channels, 1, bias=bias,
            inplace=memory_efficient, seq=seq)
        self.function = TdnnLayer(
            bn_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias,
            inplace=memory_efficient, seq=seq)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.function(x)
        return x


class DenseTdnnBlock(nn.ModuleList):

    def __init__(
            self,
            num_layers: int,
            in_channels: int,
            out_channels: int,
            bn_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: _size_2_t = 0,
            dilation: int = 1,
            bias: bool = True,
            memory_efficient: bool = False,
            seq: Tuple[str] = ('tdnn', 'relu', 'bn')
        ) -> None:
        super(DenseTdnnBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseTdnnLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                memory_efficient=memory_efficient,
                seq=seq
            )
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], 1)
        return x


class DenseTdnn(nn.Module):

    def __init__(
            self,
            feat_size: int = 80,
            xvec_size: int = 512,
            num_classes: int = None,
            growth_rate: int = 64,
            bn_size: int = 2,
            init_channels: int = 128,
            memory_efficient: bool = False
        ) -> None:
        super(DenseTdnn, self).__init__()
        bn_relu_tdnn = False
        if bn_relu_tdnn:
            seq = ('bn', 'relu', 'tdnn')
            self.xvector = nn.Sequential(OrderedDict([(
                'init_function', TimeDelay(
                    feat_size, init_channels, 5, padding=2, bias=False)
            )]))
        else:
            seq = ('tdnn', 'relu', 'bn')
            self.xvector = nn.Sequential(OrderedDict([(
                'init_function', TdnnLayer(
                    feat_size, init_channels, 5, padding=2, bias=False,
                    inplace=memory_efficient, seq=seq)
            )]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((6, 12), (3, 3), (1, 3))):
            block = DenseTdnnBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2 * dilation,
                dilation=dilation,
                bias=False,
                memory_efficient=memory_efficient,
                seq=seq
            )
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1), TdnnLayer(
                    channels, channels // 2, 1, bias=False,
                    inplace=memory_efficient, seq=seq))
            
            channels //= 2
        self.xvector.add_module('pooling', StatsPool())
        self.xvector.add_module('linear', nn.Linear(channels * 2, xvec_size))
        self.xvector.add_module('bn', nn.BatchNorm1d(xvec_size, affine=False))
        if num_classes is not None:
            self.classifier = nn.Linear(xvec_size, num_classes)


    def forward(self, x: Tensor, require_mask: bool = False) -> Tensor:
        x = x.transpose(1,2)
        x = self.xvector(x)
        if self.training:
            # shape = x.size()
            # noise = torch.FloatTensor(shape).to(device)
            # torch.randn(shape, out=noise)
            # x += noise*1e-5
            x = self.classifier(x)
        else:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
