#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0

# Reference:
# https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.common_types import _size_2_t

from ..components import TimeDelay, TdnnLayer, Tdnn2Layer, SqueezeExcite
from ..pooling import VecAttStatsPool


class Tdnn2Block(nn.Sequential):

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: _size_2_t = 0,
            dilation: int = 1,
            bias: bool = True,
            inplace: bool = False,
            scale: int = 4
        ) -> None:
        super(Tdnn2Block, self).__init__(OrderedDict([
            ('layer1', TdnnLayer(
                channels, channels, 1, bias=bias, inplace=inplace)),
            ('layer2', Tdnn2Layer(
                channels, kernel_size, stride, padding, dilation,
                bias=bias, inplace=inplace, scale=scale)),
            ('layer3', TdnnLayer(
                channels, channels, 1, bias=bias, inplace=inplace))
        ]))


class Tdnn2SeBlock(Tdnn2Block):

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: _size_2_t = 0,
            dilation: int = 1,
            bias: bool = True,
            inplace: bool = False,
            scale: int = 4,
            reduction: int = 4
        ) -> None:
        super(Tdnn2SeBlock, self).__init__(
            channels, kernel_size, stride, padding,
            dilation, bias, inplace, scale
        )
        self.add_module('se', SqueezeExcite(channels, reduction))


class EcapaTdnn(nn.Module):

    def __init__(
            self,
            feat_size: int = 80,
            xvec_size: int = 192,
            num_classes: int = None,
            channels: int = 512,
            scale: int = 8,
            reduction: int = 2
        ) -> None:
        super(EcapaTdnn, self).__init__()
        mfa_channels = channels * 3
        self.xvector = nn.Sequential(OrderedDict([
            ('init_function', TdnnLayer(
                feat_size, channels, 5, padding=2, bias=False)),
            ('block1', Tdnn2SeBlock(
                channels, 3, padding=2, dilation=2, bias=False,
                scale=scale, reduction=reduction)),
            ('block2', Tdnn2SeBlock(
                channels, 3, padding=3, dilation=3, bias=False,
                scale=scale, reduction=reduction)),
            ('block3', Tdnn2SeBlock(
                channels, 3, padding=4, dilation=4, bias=False,
                scale=scale, reduction=reduction)),
            ('mfa', TimeDelay(mfa_channels, mfa_channels, 1)),
            ('pooling', VecAttStatsPool(mfa_channels, 128)),
            ('bn1', nn.BatchNorm1d(mfa_channels * 2)),
            ('linear', nn.Linear(mfa_channels * 2, xvec_size)),
            ('bn2', nn.BatchNorm1d(xvec_size)),
        ]))
        if num_classes is not None:
            self.classifier = nn.Linear(xvec_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1,2)
        x = self.xvector.init_function(x)
        x1 = self.xvector.block1(x) + x
        x2 = self.xvector.block2(x1) + x1
        x3 = self.xvector.block3(x2) + x2
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.xvector.mfa(x))
        x = self.xvector.bn1(self.xvector.pooling(x))
        x = self.xvector.bn2(self.xvector.linear(x))
        if self.training:
            # shape = x.size()
            # noise = torch.FloatTensor(shape).to(device)
            # torch.randn(shape, out=noise)
            # x += noise*1e-5
            x = self.classifier(x)
        else:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
