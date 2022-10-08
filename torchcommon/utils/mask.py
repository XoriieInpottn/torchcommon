#!/usr/bin/env python3


import cv2 as cv
import numpy as np

__all__ = [
    'draw_mask'
]


def draw_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    mask = np.clip(mask, 0, 1) * 255
    mask = np.stack([mask, mask, np.zeros_like(mask)], 2)
    image = cv.addWeighted(image, 0.5, mask, 0.5, 0)
    return image
