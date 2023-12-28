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

from ..components import TdnnLayer, ContextAwareMask
from ..linear import TimeDelay
from ..pooling import StatsPool


class Tdnn(nn.Module):

    def __init__(
            self,
            feat_size: int = 80,
            xvec_size: int = 512,
            num_classes: int = 504,
        ) -> None:
        super(Tdnn, self).__init__()

        self.xvector = nn.Sequential(OrderedDict([
            ('layer1', TdnnLayer(feat_size, 512, 5, padding=2, bias=False)),
            ('layer2', TdnnLayer(512, 512, 3, dilation=2, padding=2, bias=False)),
            ('layer3', TdnnLayer(512, 512, 3, dilation=3, padding=3, bias=False)),
            ('transit1', TdnnLayer(512, 512, 1, bias=False)),
            ('transit2', TdnnLayer(512, 1500, 1, bias=False)),
            ('pooling', StatsPool()),
            ('linear', nn.Linear(3000, xvec_size)),
            ('bn', nn.BatchNorm1d(xvec_size, affine=False))
        ]))
        self.classifier = nn.Linear(xvec_size, num_classes)


    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1,2)
        x = self.xvector(x)
        if self.training:
            # shape = x.size()
            # noise = torch.FloatTensor(shape).to(device)
            # torch.randn(shape, out=noise)
            # x += noise*1e-5
            return self.classifier(x)
        else:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
