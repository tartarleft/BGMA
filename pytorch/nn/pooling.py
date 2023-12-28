#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0
import torch
from torch import nn, Tensor

from .functional import statistics_pooling, high_order_statistics_pooling, \
    weighted_statistics_pooling


class StatsPool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return statistics_pooling(x)


class HighOrderStatsPool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return high_order_statistics_pooling(x)


class AttStatsPool(nn.Module):

    def __init__(
            self,
            in_channels: int,
            bn_channels: int
        ) -> None:
        super(AttStatsPool, self).__init__()
        self.bn_function = nn.Conv1d(in_channels, bn_channels, 1)
        self.function = nn.Conv1d(bn_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.tanh(self.bn_function(x))
        weight = torch.softmax(self.function(out), dim=-1)
        return weighted_statistics_pooling(x, weight)


class VecAttStatsPool(nn.Module):

    def __init__(
            self,
            in_channels: int,
            bn_channels: int
        ) -> None:
        super(VecAttStatsPool, self).__init__()
        self.bn_function = nn.Conv1d(in_channels, bn_channels, 1)
        self.function = nn.Conv1d(bn_channels, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.tanh(self.bn_function(x))
        weight = torch.softmax(self.function(out), dim=-1)
        return weighted_statistics_pooling(x, weight)
