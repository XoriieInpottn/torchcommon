#!/usr/bin/env python3

import abc
import glob
import os
from typing import Union, Sequence, Any, Callable, MutableMapping

import numpy as np
from docset import DocSet, ConcatDocSet
from imgaug import SegmentationMapsOnImage
from torch.utils.data import Dataset

from .image import read_image, normalize_image

Doc = MutableMapping[str, Any]


class DocTransform(abc.ABC):

    @abc.abstractmethod
    def __call__(self, doc: Doc):
        pass


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

        self.transforms = []

    def add_transform(self, transform: Callable[[Doc], Doc]):
        self.transforms.append(transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        doc = self.ds[i]
        for transform in self.transforms:
            doc = transform(doc)
        return doc


class ImageDataset(Dataset):

    def __init__(
            self,
            data_source,
            image_field='image',
            bbox_field=None,
            mask_field=None,
            augmenter=None,
            normalize=True,
            transpose=True
    ) -> None:
        super(ImageDataset, self).__init__()
        self.data_source = data_source
        self.image_field = image_field
        self.bbox_field = bbox_field
        self.mask_field = mask_field
        self.augmenter = augmenter
        self.normalize = normalize
        self.transpose = transpose

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        doc = self.data_source[idx]
        image = read_image(doc[self.image_field])

        aug_args = {'image': image}
        if self.bbox_field:
            pass
        if self.mask_field:
            mask = doc[self.mask_field]
            assert isinstance(mask, np.ndarray)
            mask_rank = len(mask.shape)
            assert mask_rank == 2 or mask_rank == 3
            aug_args['segmentation_maps'] = SegmentationMapsOnImage(doc[self.mask_field], image.shape)

        aug_result = self.augmenter(aug_args)
        image = next(aug_result)
        if self.bbox_field:
            pass
        if self.mask_field:
            maps_oi = next(aug_result)
            mask = maps_oi.arr.squeeze(-1)
            doc[self.mask_field] = mask

        if self.normalize:
            image = normalize_image(image, transpose=self.transpose)
        doc[self.image_field] = image
        return doc
