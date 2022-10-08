#!/usr/bin/env python3

from typing import Tuple

import cv2 as cv
import numpy as np
from imgaug import BoundingBoxesOnImage, BoundingBox

__all__ = [
    'xywh_to_xyxy',
    'xyxy_to_xywh',
    'pct_to_pixel',
    'pixel_to_pct',
    'array_to_bboxes_on_image',
    'bboxes_on_image_to_array'
]


def xywh_to_xyxy(a: np.ndarray, inplace=False):
    if not inplace:
        a = np.array(a)
    x, y = a[..., 0:1], a[..., 1:2]
    ow, oh = a[..., 2:3] * 0.5, a[..., 3:4] * 0.5
    x1, y1, x2, y2 = x - ow, y - oh, x + ow, y + oh
    a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4] = x1, y1, x2, y2
    return a


def xyxy_to_xywh(a: np.ndarray, inplace=False):
    if not inplace:
        a = np.array(a)
    x1, y1, x2, y2 = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    x, y, w, h = (x1 + x2) * 0.5, (y1 + y1) * 0.5, x2 - x1, y2 - y1
    a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4] = x, y, w, h
    return a


def pct_to_pixel(a: np.ndarray, height, width, inplace=False):
    if not inplace:
        a = np.array(a)
    a[..., 0:1] *= width
    a[..., 1:2] *= height
    a[..., 2:3] *= width
    a[..., 3:4] *= height
    return a


def pixel_to_pct(a: np.ndarray, height, width, inplace=False):
    if not inplace:
        a = np.array(a)
    a[..., 0:1] /= width
    a[..., 1:2] /= height
    a[..., 2:3] /= width
    a[..., 3:4] /= height
    return a


def array_to_bboxes_on_image(
        arr: np.ndarray,
        shape: Tuple[int, int],
        data_format='xyxy',
        pct=False
) -> BoundingBoxesOnImage:
    if data_format == 'xyxy':
        if pct:
            arr = pct_to_pixel(arr, shape[0], shape[1])
    elif data_format == 'xywh':
        arr = xywh_to_xyxy(arr)
        if pct:
            arr = pct_to_pixel(arr, shape[0], shape[1], inplace=True)
    else:
        raise RuntimeError(f'Unsupported data format "{data_format}".')
    return BoundingBoxesOnImage([BoundingBox(*bbox[0:4], bbox[4]) for bbox in arr], shape)


def bboxes_on_image_to_array(
        bboxes_on_image: BoundingBoxesOnImage,
        data_format='xyxy',
        pct=False
) -> np.ndarray:
    n = len(bboxes_on_image.bounding_boxes)
    arr = np.zeros((n, 5), dtype=np.float32)
    for i, bbox in enumerate(bboxes_on_image.bounding_boxes):
        if data_format == 'xyxy':
            arr[i] = bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.label
        elif data_format == 'xywh':
            arr[i] = bbox.center_x, bbox.center_y, bbox.width, bbox.height, bbox.label
    if pct:
        height, width = bboxes_on_image.shape
        arr = pixel_to_pct(arr, height, width, inplace=True)
    return arr


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
