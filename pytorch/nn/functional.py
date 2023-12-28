#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0
import torch
from torch import Tensor


def statistics_pooling(
        x: Tensor,
        dim: int = -1,
        keepdim: bool = False,
        unbiased: bool = True
    ) -> Tensor:
    mean = torch.mean(x, dim=dim)
    std = torch.std(x, dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def high_order_statistics_pooling(
        x: Tensor,
        dim: int = -1,
        keepdim: bool = False,
        unbiased: bool = True,
        eps: float = 1e-2
    ) -> Tensor:
    mean = torch.mean(x, dim=dim)
    std = torch.std(x, dim=dim, unbiased=unbiased)
    submean = x - mean.unsqueeze(dim=dim)
    norm = submean / std.clamp(min=eps).unsqueeze(dim=dim)
    skewness = torch.mean(torch.pow(norm, 3), dim=dim)
    kurtosis = torch.mean(torch.pow(norm, 4), dim=dim)
    stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def weighted_statistics_pooling(
        x: Tensor,
        weight: Tensor,
        dim: int = -1,
        keepdim: bool = False,
        eps: float = 1e-8
    ) -> Tensor:
    mean = torch.sum(x * weight, dim=dim)
    var = torch.sum(x ** 2 * weight, dim=dim) - mean ** 2
    std = torch.sqrt(var.clamp(min=eps))
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def margin(
        x: Tensor,
        target: Tensor,
        angle: float = 0,
        cosine: float = 0
    ) -> Tensor:
    indices = range(target.shape[0])
    if angle != 0:
        x[indices, target] = torch.cos(torch.acos(x[indices, target]) + angle)
    if cosine != 0:
        x[indices, target] += - cosine
    return x
