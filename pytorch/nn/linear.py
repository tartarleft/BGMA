#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _single, _pair


class NormLinear(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int
        ) -> None:
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(x)
        w = F.normalize(self.weight)
        return F.linear(x, w)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class TimeDelay(nn.Module):
    # We implement time delay neural network in two ways,
    # including conv (nn.Conv1d) and linear (nn.Linear).
    # Linear supports different paddings in two sides.
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: _size_2_t = 0,
            dilation: int = 1,
            bias: bool = True,
            impl: str = 'conv'
        ) -> None:
        super(TimeDelay, self).__init__()
        if impl not in ['conv', 'linear']:
            raise ValueError('impl must be conv or linear')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.dilation = _single(dilation)
        if impl == 'conv':
            self.padding = _single(padding)
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels, kernel_size))
        else:
            self.padding = _pair(padding)
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels * kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.impl = impl
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.impl == 'conv':
            return F.conv1d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation)
        else:
            x = F.pad(x, self.padding).unsqueeze(1)
            x = F.unfold(
                x, (self.in_channels,)+self.kernel_size,
                dilation=(1,)+self.dilation, stride=(1,)+self.stride)
            return F.linear(
                x.transpose(1, 2), self.weight, self.bias).transpose(1, 2)

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, '
             'stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
