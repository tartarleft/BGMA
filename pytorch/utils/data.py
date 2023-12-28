#!/usr/bin/env python3

# Copyright   2020-2021  Nanjing University (Author: Ya-Qi Yu)
# Apache 2.0
import os
import random
from typing import Tuple

import kaldiio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class KaldiFeatDataset(Dataset):

    def __init__(
            self,
            root: str = None,
            transforms: Tuple[object] = None,
            label: str = 'utt'
        ) -> None:
        super(KaldiFeatDataset, self).__init__()
        self.transforms = transforms
        self.feats = []
        self.utt2sid = None
        spk2sid = {}
        cnt = 0
        if label == 'sid':
            self.utt2sid = {}
            with open(os.path.join(root, 'utt2spk'), 'r') as f:
                for line in f:
                    utt, spk = line.split()
                    if spk2sid.get(spk) is None:
                        spk2sid[spk] = cnt
                        cnt += 1
                    self.utt2sid[utt] = spk2sid[spk]
        with open(os.path.join(root, 'feats.scp'), 'r') as f:
            for line in f:
                utt, ark = line.split()
                self.feats.append((ark, utt))

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        ark, utt = self.feats[index]
        a = kaldiio.load_mat(ark)
        if self.transforms is not None:
            for t in self.transforms:
                a = t(a)
        if self.utt2sid is not None:
            sid = self.utt2sid[utt]
            return a, sid
        return a, utt


class Transpose:

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return a.transpose((1, 0))


# Note: Check https://github.com/yuyq96/kaldifeat/blob/master/feature.py
# for more information.
class CMVN:

    def __init__(
            self,
            center: bool = False,
            window: int = 600,
            min_window: int = 100,
            norm_vars: bool = False
        ) -> None:
        self.center = center
        self.window = window
        self.min_window = min_window
        self.norm_vars = norm_vars

    def __call__(self, a: np.ndarray) -> np.ndarray:
        frames, feat_size = a.shape
        std = 1
        if self.center:
            if frames <= self.window:
                mean = a.mean(axis=0, keepdims=True).repeat(frames, axis=0)
                if self.norm_vars:
                    std = a.std(axis=0, keepdims=True).repeat(frames, axis=0)
            else:
                lwin = self.window // 2
                rwin = (self.window - 1) // 2
                a1 = a[:self.window]
                a2 = sliding_window(a.T, self.window, 1)
                a3 = a[-self.window:]
                mean1 = a1.mean(axis=0, keepdims=True).repeat(lwin, axis=0)
                mean2 = a2.mean(axis=2).T
                mean3 = a3.mean(axis=0, keepdims=True).repeat(rwin, axis=0)
                mean = np.concatenate([mean1, mean2, mean3])
                if self.norm_vars:
                    std1 = a1.std(axis=0, keepdims=True).repeat(lwin, axis=0)
                    std2 = a2.std(axis=2).T
                    std3 = a3.mean(axis=0, keepdims=True).repeat(rwin, axis=0)
                    std = np.concatenate([std1, std2, std3])
        else:
            if frames <= self.min_window:
                mean = a.mean(axis=0, keepdims=True).repeat(frames, axis=0)
                if self.norm_vars:
                    std = a.std(axis=0, keepdims=True).repeat(frames, axis=0)
            else:
                a1 = a[:self.min_window]
                mean1 = a1.mean(axis=0, keepdims=True).repeat(
                    self.min_window, axis=0)
                a2_cumsum = np.cumsum(a[:self.window], axis=0)[self.min_window:]
                cumcnt = np.arange(
                    self.min_window + 1, min(self.window, frames) + 1,
                    dtype=a.dtype)[:, np.newaxis]
                mean2 = a2_cumsum / cumcnt
                mean = np.concatenate([mean1, mean2])
                if self.norm_vars:
                    std1 = a1.std(axis=0, keepdims=True).repeat(
                        self.min_window, axis=0)
                    feat2_power_cumsum = np.cumsum(np.square(
                        a[:self.window]), axis=0)[self.min_window:]
                    std2 = np.sqrt(
                        feat2_power_cumsum / cumcnt - np.square(mean2))
                    std = np.concatenate([std1, std2])
                if frames > self.window:
                    a3 = sliding_window(a.T, self.window, 1)
                    mean3 = a3.mean(axis=2).T
                    mean = np.concatenate([mean, mean3[1:]])
                    if self.norm_vars:
                        std3 = a3.std(axis=2).T
                        std = np.concatenate([std, std3[1:]])
        a = (a - mean) / std
        return a


def _get_random_offset(total_len, sample_len):
    if sample_len > total_len:
        sys.exit("code error: sample_len > total_len")
    free_len = total_len - sample_len
    offset = random.randint(0, free_len)
    return offset


class Crop:

    def __init__(
            self,
            frames: int = 200
        ) -> None:
        self.frames = frames
    
    def __call__(self, a: np.ndarray) -> np.ndarray:
        if a.shape[1] > self.frames:
            offset = _get_random_offset(a.shape[1], self.frames)
            a = a[:, offset:offset + self.frames]
        else:
            left = self.frames
            tmp = []
            while left > 0:
                l = min(a.shape[1], left)
                tmp.append(a[:, :l])
                left -= l
            a = np.hstack(tmp)
        return a


def spec_mask(a, size, axis):
    mask = np.random.randint(0, size + 1)
    offset = np.random.randint(0, a.shape[axis] - mask + 1)
    if axis == 0:
        a[offset:offset+mask] = 0
        inverted_factor = a.shape[0] / (a.shape[0] - mask)
        a *= inverted_factor
    elif axis == 1:
        a[:, offset:offset+mask] = 0
    else:
        raise ValueError('Only Frequency and Time masking are supported')
    return a


class SpecAug:

    def __init__(self, size=(10, 5), repeats=(1, 1)):
        super(SpecAug, self).__init__()
        self.size = size
        self.repeats = repeats

    def __call__(self, a):
        for i in range(2):
            if self.size[i] > 0:
                for _ in range(self.repeats[i]):
                    a = spec_mask(a, self.size[i], i)
        return a
