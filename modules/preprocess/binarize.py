"""
binarize.py
===========
Stage 2c: Adaptive thresholding and border/frame removal for engineering
drawings. Adaptive (local) thresholding handles uneven scan illumination
far better than a single global threshold.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale; no-op if already single-channel."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adaptive_binarize(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive (Gaussian) thresholding to produce a clean black-on-white
    binary image suitable for contour/edge analysis.
    """
    cfg = settings.preprocess
    gray = to_grayscale(image)

    block_size = cfg.adaptive_thresh_block_size
    if block_size % 2 == 0:
        block_size += 1  # cv2 requires an odd block size

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        cfg.adaptive_thresh_c,
    )
    logger.debug("Adaptive binarization applied with block_size=%d, C=%d", block_size, cfg.adaptive_thresh_c)
    return binary


def remove_border(image: np.ndarray, border_px: int | None = None) -> np.ndarray:
    """Crop a fixed-width border/frame that scanners often introduce."""
    cfg = settings.preprocess
    px = border_px if border_px is not None else cfg.border_crop_px
    h, w = image.shape[:2]
    px = max(0, min(px, min(h, w) // 4))
    if px == 0:
        return image
    return image[px: h - px, px: w - px]


def remove_title_block_noise(binary_image: np.ndarray) -> np.ndarray:
    """
    Light morphological cleanup to remove single-pixel speckle noise that
    commonly appears near scanned title blocks / borders, without eroding
    legitimate thin line-work.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned
