#!/usr/bin/env python3


import random
from typing import Tuple, Union

import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
from imgaug import BoundingBoxesOnImage

__all__ = [
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'read_image',
    'normalize_image',
    'denormalize_image',
    'hwc_to_chw',
    'chw_to_hwc',
    'plot_bboxes',
    'draw_mask',
    'ResizedCrop',
    'ColorJitter',
    'Mosaic'
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def read_image(path_or_data):
    if isinstance(path_or_data, str):
        image = cv.imread(path_or_data, cv.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to load image {path_or_data}')
    elif isinstance(path_or_data, bytes):
        image = cv.imdecode(np.frombuffer(path_or_data, np.byte), cv.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Failed to load image')
    else:
        raise RuntimeError(f'Invalid input type {type(path_or_data)}.')
    cv.cvtColor(image, cv.COLOR_BGR2RGB, image)
    return image


def normalize_image(
        image: np.ndarray,
        mean: Union[np.ndarray, float] = IMAGENET_MEAN,
        std: Union[np.ndarray, float] = IMAGENET_STD,
        transpose=False
) -> np.ndarray:
    image = np.array(image, dtype=np.float32)
    image -= mean
    image /= std
    if transpose:
        image = hwc_to_chw(image)
    return image


def denormalize_image(
        image: np.ndarray,
        mean: Union[np.ndarray, float] = IMAGENET_MEAN,
        std: Union[np.ndarray, float] = IMAGENET_STD,
        transpose=False
) -> np.ndarray:
    if transpose:
        image = chw_to_hwc(image)
    image *= std
    image += mean
    np.clip(image, 0, 255, out=image)
    image = np.array(image, dtype=np.uint8)
    return image


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    if len(image.shape) != 3:
        raise RuntimeError('Image should be a 3-dimensional tensor/ndarray.')
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image)
    return image


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    if len(image.shape) != 3:
        raise RuntimeError('Image should be a 3-dimensional tensor/ndarray.')
    image = np.transpose(image, (1, 2, 0))
    image = np.ascontiguousarray(image)
    return image


def plot_bboxes(
        image: np.ndarray,
        bboxes: np.ndarray,
        num_classes: int,
        cmap: str = 'Spectral',
        class_map: dict = None
) -> np.ndarray:
    image = np.copy(image)
    boxes, label = bboxes[:, :4], bboxes[:, 4]
    height, width, _ = image.shape
    boxes = boxes * (width, height, width, height)
    boxes = boxes.astype(np.int64)
    label = label.astype(np.int64)

    from matplotlib import cm
    cmap = cm.get_cmap(cmap)

    for (x, y, w, h), l in zip(boxes, label):
        color = np.array(cmap(l / num_classes), dtype=np.float32) * 255
        color = [int(c) for c in color]
        name = 'object' if class_map is None else class_map[l]

        ow, oh = w // 2, h // 2
        x1, y1, x2, y2 = x - ow, y - oh, x + ow, y + oh
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv.getTextSize(name, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p = (x1, y1 - text_size[1])
        p1 = (p[0] - 2 // 2, p[1] - 2 - baseline)
        p2 = (p[0] + text_size[0], p[1] + text_size[1])
        cv.rectangle(image, p1, p2, color, -1)
        cv.putText(image, name, (p[0], p[1] + baseline), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    return image


def draw_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    mask = np.clip(mask, 0, 1) * 255
    mask = np.stack([mask, mask, np.zeros_like(mask)], 2)
    image = cv.addWeighted(image, 0.5, mask, 0.5, 0)
    return image


class ResizedCrop(iaa.Sequential):

    def __init__(
            self,
            width: int,
            height: int,
            scale: float = 1.0,
            ratio: float = 1.33,
            interpolation='linear'
    ) -> None:
        assert scale >= 1.0, f'Invalid scale {scale}. It should >= 1.'
        assert ratio > 0, f'Invalid ratio {ratio}. It should > 0.'
        if ratio < 1.0:
            ratio = 1.0 / ratio
        min_width = int(width * scale)
        max_width = int(min_width * ratio)
        min_height = int(height * scale)
        max_height = int(min_height * ratio)
        super(ResizedCrop, self).__init__([
            iaa.Resize(
                {'width': (min_width, max_width), 'height': (min_height, max_height)},
                interpolation=interpolation
            ),
            iaa.CropToFixedSize(width=width, height=height),
        ])


class ColorJitter(iaa.Sequential):

    def __init__(
            self,
            hue_shift: Union[float, Tuple[float], None] = 0.05,
            saturation_factor: Union[float, Tuple[float], None] = 0.2,
            brightness_factor: Union[float, Tuple[float], None] = 0.2,
            contrast_factor: Union[float, Tuple[float], None] = 0.2
    ) -> None:
        """Randomly change the hue, saturation, brightness and contrast of an image.

        Args:
            hue_shift: How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            saturation_factor: How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            brightness_factor: How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast_factor: How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
        """
        if isinstance(hue_shift, float):
            h = (-int(hue_shift * 255), int(hue_shift * 255))
        elif isinstance(hue_shift, (tuple, list)) and len(hue_shift) == 2:
            h = (int(hue_shift[0] * 255), int(hue_shift[1] * 255))
        elif hue_shift is None:
            h = None
        else:
            raise RuntimeError(f'Invalid hue_shift {hue_shift}.')

        if isinstance(saturation_factor, float):
            s = (max(1.0 - saturation_factor, 0), 1.0 + saturation_factor)
        elif isinstance(saturation_factor, (tuple, list)) and len(saturation_factor) == 2:
            s = saturation_factor
        elif saturation_factor is None:
            s = None
        else:
            raise RuntimeError(f'Invalid saturation_factor {saturation_factor}.')

        if isinstance(brightness_factor, float):
            v = (max(1.0 - brightness_factor, 0), 1.0 + brightness_factor)
        elif isinstance(brightness_factor, (tuple, list)) and len(brightness_factor) == 2:
            v = brightness_factor
        elif brightness_factor is None:
            v = None
        else:
            raise RuntimeError(f'Invalid brightness_factor {brightness_factor}.')

        if isinstance(contrast_factor, float):
            c = (max(1.0 - contrast_factor, 0), 1.0 + contrast_factor)
        elif isinstance(contrast_factor, (tuple, list)) and len(contrast_factor) == 2:
            c = contrast_factor
        elif contrast_factor is None:
            c = None
        else:
            raise RuntimeError(f'Invalid contrast_factor {contrast_factor}.')

        super(ColorJitter, self).__init__([
            iaa.WithColorspace(
                from_colorspace=iaa.CSPACE_RGB,
                to_colorspace=iaa.CSPACE_HSV,
                children=iaa.Sequential([
                    iaa.WithChannels(0, iaa.Add(h)) if h else iaa.Identity(),
                    iaa.WithChannels(1, iaa.Multiply(s)) if s else iaa.Identity(),
                    iaa.WithChannels(2, iaa.Multiply(v)) if v else iaa.Identity()
                ])
            ),
            iaa.LinearContrast(c) if c else iaa.Identity()
        ])


class Mosaic(iaa.Augmenter):

    def __init__(self, image_size, shrink=0.1, cval=0.0):
        super(Mosaic, self).__init__()
        self.cval = cval
        s = image_size
        c = shrink
        self.aug_list = [
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=(0, (0, c), (0, c), 0), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='left-top', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='right-bottom', pad_cval=cval)
            ]),
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=(0, 0, (0, c), (0, c)), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='right-top', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='left-bottom', pad_cval=cval)
            ]),
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=((0, c), (0, c), 0, 0), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='left-bottom', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='right-top', pad_cval=cval)
            ]),
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=((0, c), 0, 0, (0, c)), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='right-bottom', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='left-top', pad_cval=cval)
            ])
        ]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        kwargs_list = [{} for _ in range(4)]
        result_dict = {}

        if batch.images is not None:
            image_list = batch.images
            assert len(image_list) == 4
            for i in range(4):
                kwargs_list[i]['image'] = image_list[i]
                result_dict['image'] = []

        if batch.bounding_boxes is not None:
            bboi_list = batch.bounding_boxes
            assert len(bboi_list) == 4
            for i in range(4):
                kwargs_list[i]['bounding_boxes'] = bboi_list[i]
                result_dict['bounding_boxes'] = []

        aug_list = list(self.aug_list)
        random.shuffle(aug_list)
        for i in range(4):
            ret = aug_list[i](**kwargs_list[i])
            if not isinstance(ret, tuple):
                result_dict['image'].append(ret)
            else:
                for result, name in zip(ret, result_dict):
                    result_dict[name].append(result)

        if 'image' in result_dict:
            image_list = result_dict['image']
            image = np.zeros_like(image_list[0], dtype=np.int16)
            for i in range(0, 4):
                image += image_list[i]
            image -= int(3 * self.cval)
            image = np.clip(image, 0, 255).astype(np.uint8)
            batch.images = [image]

        if 'bounding_boxes' in result_dict:
            bboi_list = result_dict['bounding_boxes']
            bbox_list = []
            for bboi in bboi_list:
                bbox_list.extend(bboi.bounding_boxes)
            batch.bounding_boxes = [BoundingBoxesOnImage(
                bounding_boxes=bbox_list,
                shape=bboi_list[0].shape
            )]

        return batch

    def __call__(self, *args, **kwargs):
        ret = super(Mosaic, self).__call__(*args, **kwargs)
        if isinstance(ret, tuple):
            return tuple(value[0] for value in ret)
        else:
            return ret[0]

    def get_parameters(self):
        params = []
        for aug in self.aug_list:
            params.extend(aug.get_parameters())
        return params
