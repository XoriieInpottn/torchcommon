#!/usr/bin/env python3

import bisect
import collections
import glob
import os
import random
from typing import Iterable, Mapping, Callable, MutableMapping
from typing import Union, Sequence

import numpy as np
import torch
from docset import DocSet, ConcatDocSet
from imgaug import SegmentationMapsOnImage, BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from torch.utils.data import Dataset, ConcatDataset

from .image import read_image, normalize_image


class DSDataset(Dataset):

    def __init__(self, path: Union[str, Sequence[str]]):
        path_list = []
        if isinstance(path, str):
            if os.path.isdir(path):
                path_list.extend(glob.iglob(os.path.join(path, '*.ds')))
            else:
                path_list.append(path)
        else:
            path_list.extend(path)

        self.ds_list = [DocSet(path) for path in path_list]
        self.ds = self.ds_list[0] if len(self.ds_list) == 1 else ConcatDocSet(self.ds_list)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


class ImageDataset(Dataset):

    def __init__(
            self,
            data_source,
            image_field='image',
            bbox_field=None,
            mask_field=None,
            pre_augment: Callable[[MutableMapping], None] = None,
            augmenter: iaa.Augmenter = None,
            post_augment: Callable[[MutableMapping], None] = None,
            normalize=True,
            transpose=True
    ) -> None:
        super(ImageDataset, self).__init__()
        self.data_source = data_source
        self.image_field = image_field
        self.bbox_field = bbox_field
        self.mask_field = mask_field
        self.pre_augment = pre_augment
        self.augmenter = augmenter
        self.post_augment = post_augment
        self.normalize = normalize
        self.transpose = transpose

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        doc = self.data_source[idx]
        doc[self.image_field] = read_image(doc[self.image_field])

        if callable(self.pre_augment):
            self.pre_augment(doc)
        self._apply_augmenter(doc)
        if callable(self.post_augment):
            self.post_augment(doc)

        if self.normalize:
            doc[self.image_field] = normalize_image(doc[self.image_field], transpose=self.transpose)
        return doc

    def _apply_augmenter(self, doc):
        if self.augmenter is None:
            return

        if not isinstance(self.augmenter, iaa.Augmenter):
            raise RuntimeError('Invalid augmenter.')

        ################################################################################
        # prepare augmenter arguments
        ################################################################################
        aug_args = {'image': doc[self.image_field]}

        if self.bbox_field:
            image = doc[self.image_field]
            bboxes = doc[self.bbox_field]
            bbox_objs = []
            for bbox in bboxes:
                # x, y, w, h, label = bbox
                # x, y, w, h = x * iw, y * ih, w * iw, h * ih
                # ow, oh = w * 0.5, h * 0.5
                # x1, y1, x2, y2 = x - ow, y - oh, x + ow, y + oh
                x1, y1, x2, y2, label = bbox
                bbox_obj = BoundingBox(x1, y1, x2, y2, label)
                bbox_objs.append(bbox_obj)
            aug_args['bounding_boxes'] = BoundingBoxesOnImage(
                bounding_boxes=bbox_objs,
                shape=image.shape
            )

        if self.mask_field:
            image = doc[self.image_field]
            mask = doc[self.mask_field]
            assert isinstance(mask, np.ndarray)
            mask_rank = len(mask.shape)
            assert mask_rank == 2 or mask_rank == 3
            aug_args['segmentation_maps'] = SegmentationMapsOnImage(
                arr=mask,
                shape=image.shape
            )

        ################################################################################
        # perform augmentation
        ################################################################################
        aug_result = self.augmenter(**aug_args)
        if len(aug_args) == 1:
            aug_result = [aug_result]

        ################################################################################
        # collect augmentation results
        ################################################################################
        aug_result = iter(aug_result)
        doc[self.image_field] = next(aug_result)

        if self.bbox_field:
            bbox_objs = next(aug_result)
            bbox_objs = bbox_objs.remove_out_of_image_fraction(0.8).clip_out_of_image()
            bboxes = np.empty((len(bbox_objs), 5), dtype=np.float32)
            for i, bbox_obj in enumerate(bbox_objs):
                x1, y1, x2, y2, label = bbox_obj.x1, bbox_obj.y1, bbox_obj.x2, bbox_obj.y2, bbox_obj.label
                # x, y, w, h = (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1
                # x, y, w, h = x / iw, y / ih, w / iw, h / ih
                # bboxes[i] = x, y, w, h, label
                bboxes[i] = x1, y1, x2, y2, label
            doc[self.bbox_field] = bboxes

        if self.mask_field:
            maps_oi = next(aug_result)
            mask = maps_oi.arr.squeeze(-1)
            doc[self.mask_field] = mask


def merge_docs(doc_list):
    if isinstance(doc_list[0], Mapping):
        merged = collections.defaultdict(list)
        for doc in doc_list:
            for name, value in doc.items():
                merged[name].append(value)
        result = {}
        for name, values in merged.items():
            try:
                if isinstance(values[0], np.ndarray):
                    values = np.stack(values)
                elif isinstance(values[0], torch.Tensor):
                    values = torch.stack(values)
            except ValueError:
                pass
            result[name] = values
        return result
    else:
        return doc_list


class KShotDataset(ConcatDataset):

    def __init__(
            self,
            data_sources: Iterable[Dataset],
            num_shots: int,
            merge_fn=merge_docs
    ) -> None:
        assert num_shots > 0
        for i, data_source in enumerate(data_sources):
            if len(data_source) < (num_shots + 1):  # type: ignore
                raise RuntimeError(f'The {i}-th dataset is smaller than {num_shots}.')
        super(KShotDataset, self).__init__(data_sources)
        self.num_shots = num_shots
        self.merge_fn = merge_fn

    def __getitem__(self, idx):
        dataset_idx, query_idx = self._parse_idx(idx)
        dataset = self.datasets[dataset_idx]
        query_doc = dataset[query_idx]

        supp_docs = []
        selected = {query_idx}
        supp_idx = query_idx
        largest_idx = len(dataset) - 1  # type: ignore
        for _ in range(self.num_shots):
            while supp_idx in selected:
                supp_idx = random.randint(0, largest_idx)
            selected.add(supp_idx)
            supp_docs.append(dataset[supp_idx])

        if callable(self.merge_fn):
            supp_docs = self.merge_fn(supp_docs)

        return query_doc, supp_docs

    def _parse_idx(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx
