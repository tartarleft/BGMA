#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0
from torch import nn, Tensor

from .functional import margin


class Margin(nn.Module):

    def __init__(
            self,
            angle: float = 0,
            cosine: float = 0
        ) -> None:
        super(Margin, self).__init__()
        self.angle = angle
        self.cosine = cosine

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return margin(x, target, self.angle, self.cosine)

    def extra_repr(self) -> str:
        return 'angle={}, cosine={}'.format(
            self.angle, self.cosine
        )


class Scaling(nn.Module):

    def __init__(
            self,
            factor: float = 32.0
        ) -> None:
        super(Scaling, self).__init__()
        self.factor = factor

    # Note: target is a placeholder for future support
    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        return x * self.factor

    def extra_repr(self) -> str:
        return 'factor={}'.format(
            self.factor
        )


def amsoftmax(margin, scaling_factor):
    return Margin(cosine=margin), Scaling(scaling_factor)


def aamsoftmax(margin, scaling_factor):
    return Margin(angle=margin), Scaling(scaling_factor)
