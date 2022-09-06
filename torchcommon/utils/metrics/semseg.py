#!/usr/bin/env python3

"""
@author: xi
@since: 2021-07-21
"""

import collections
from typing import List, Tuple, Union, Sequence

import cv2 as cv
import numpy as np

__all__ = [
    'IouMeter',
]


class IouMeter(object):

    def __init__(
            self,
            ignore_class: int = None,
            bg_class: int = 0,
            eps=1e-10
    ) -> None:
        self._ignore_class = ignore_class
        self._bg_class = bg_class
        self._eps = eps

        self._inter_fg = 0
        self._union_fg = 0
        self._inter_bg = 0
        self._union_bg = 0
        self._pred = 0
        self._true = 0
        self._inter_dict = collections.defaultdict(int)
        self._union_dict = collections.defaultdict(int)

    def update(
            self,
            output: np.ndarray,
            target: np.ndarray,
            label: Union[np.ndarray, Sequence[int], int] = None,
            size_list: List[Tuple[int, int]] = None
    ) -> None:
        """Update the meter's state by a batch of result.

        Args:
            output: dtype=int64, shape=(n, h, w)
            target: dtype=int64, shape=(n, h, w)
            label: list of classes
            size_list: list of mask size
        """
        assert len(output) == len(target)
        for i, (output_i, target_i) in enumerate(zip(output, target)):
            output_i = output_i.copy()  # if you don't copy, you will corrupt the original input
            target_i = target_i.copy()  # if you don't copy, you will corrupt the original input

            # Scale the results to a given size.
            if size_list is not None:
                output_i = np.array(output_i, dtype=np.uint8)
                output_i = cv.resize(output_i, size_list[i], interpolation=cv.INTER_NEAREST)
                output_i = np.array(output_i, dtype=np.int64)
                target_i = np.array(target_i, dtype=np.uint8)
                target_i = cv.resize(target_i, size_list[i], interpolation=cv.INTER_NEAREST)
                target_i = np.array(target_i, dtype=np.int64)

            # Re-assign a labels to a foreground-background segmentation results.
            if label is not None:
                label_i = label if isinstance(label, int) else label[i]
                output_i[output_i == 1] = label_i
                target_i[target_i == 1] = label_i

            self._update(output_i, target_i)

    def _update(self, output_i: np.ndarray, target_i: np.ndarray):
        if self._ignore_class is not None:
            output_i[np.where(target_i == self._ignore_class)] = self._ignore_class

        c_set = set()
        for c in np.unique(output_i):
            c_set.add(int(c))
        for c in np.unique(target_i):
            c_set.add(int(c))
        if self._bg_class in c_set:
            c_set.remove(self._bg_class)
        if self._ignore_class in c_set:
            c_set.remove(self._ignore_class)

        for c in c_set:
            inter = ((output_i == c) & (target_i == c)).sum()
            union = ((output_i == c) | (target_i == c)).sum()
            self._inter_dict[c] += inter
            self._union_dict[c] += union
            self._inter_fg += inter
            self._union_fg += union

            pred = (output_i == c).sum()
            true = (target_i == c).sum()
            self._pred += pred
            self._true += true

        inter_bg = ((output_i == self._bg_class) & (target_i == self._bg_class)).sum()
        union_bg = ((output_i == self._bg_class) | (target_i == self._bg_class)).sum()
        self._inter_bg += inter_bg
        self._union_bg += union_bg

    def m_iou(self):
        iou_list = []
        for clazz in self._inter_dict:
            inter = self._inter_dict[clazz]
            union = self._union_dict[clazz]
            iou_list.append(inter / (union + self._eps))
        return np.mean(iou_list)

    def fb_iou(self):
        iou_fg = self._inter_fg / (self._union_fg + self._eps)
        iou_bg = self._inter_bg / (self._union_bg + self._eps)
        return (iou_fg + iou_bg) * 0.5

    def precision(self):
        return self._inter_fg / (self._pred + self._eps)

    def recall(self):
        return self._inter_fg / (self._true + self._eps)


class IouMeterBak(object):

    def __init__(self, ignore_class: int = None, eps=1e-10):
        self._ignore_class = ignore_class
        self._eps = eps

        self._inter_fg = 0
        self._union_fg = 0
        self._inter_bg = 0
        self._union_bg = 0
        self._pred = 0
        self._true = 0
        self._inter_dict = collections.defaultdict(int)
        self._union_dict = collections.defaultdict(int)

    def update(
            self,
            output: np.ndarray,
            target: np.ndarray,
            label: Union[np.ndarray, Sequence[int], int],
            size_list: List[Tuple[int, int]] = None
    ) -> None:
        """Update the meter's state by a batch of result.

        Args:
            output: dtype=int64, shape=(n, h, w)
            target: dtype=int64, shape=(n, h, w)
            label: list of classes
            size_list: list of mask size
        """
        if isinstance(label, int):
            label = np.full((output.shape[0],), label, dtype=np.int64)
        assert len(output) == len(target) == len(label)
        for i, (output_i, target_i, label_i) in enumerate(zip(output, target, label)):
            output_i = output_i.copy()  # if you don't copy, you will corrupt the original input
            label_i = int(label_i)
            if size_list is not None:
                output_i = np.array(output_i, dtype=np.uint8)
                output_i = cv.resize(output_i, size_list[i], interpolation=cv.INTER_NEAREST)
                output_i = np.array(output_i, dtype=np.int64)
                target_i = np.array(target_i, dtype=np.uint8)
                target_i = cv.resize(target_i, size_list[i], interpolation=cv.INTER_NEAREST)
                target_i = np.array(target_i, dtype=np.int64)

            if self._ignore_class is not None:
                output_i[np.where(target_i == self._ignore_class)] = self._ignore_class

            inter = ((output_i == 1) & (target_i == 1)).sum()
            union = ((output_i == 1) | (target_i == 1)).sum()
            self._inter_dict[label_i] += inter
            self._union_dict[label_i] += union
            self._inter_fg += inter
            self._union_fg += union

            pred = (output_i == 1).sum()
            true = (target_i == 1).sum()
            self._pred += pred
            self._true += true

            inter_bg = ((output_i == 0) & (target_i == 0)).sum()
            union_bg = ((output_i == 0) | (target_i == 0)).sum()
            self._inter_bg += inter_bg
            self._union_bg += union_bg

    def m_iou(self):
        iou_list = []
        for clazz in self._inter_dict:
            inter = self._inter_dict[clazz]
            union = self._union_dict[clazz]
            iou_list.append(inter / (union + self._eps))
        return np.mean(iou_list)

    def fb_iou(self):
        iou_fg = self._inter_fg / (self._union_fg + self._eps)
        iou_bg = self._inter_bg / (self._union_bg + self._eps)
        return (iou_fg + iou_bg) * 0.5

    def precision(self):
        return self._inter_fg / (self._pred + self._eps)

    def recall(self):
        return self._inter_fg / (self._true + self._eps)
