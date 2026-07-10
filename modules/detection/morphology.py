"""
morphology.py
=============
Morphological opening/closing operations to clean up fused difference
masks: opening removes small speckle noise, closing bridges small gaps
between fragments of the same real change (e.g. broken dimension lines).
"""

from __future__ import annotations

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Apply opening (noise removal) followed by closing (gap bridging) to a
    binary difference mask.
    """
    cfg = settings.detection
    k = cfg.morph_kernel_size
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iterations
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iterations
    )

    logger.debug(
        "Morphology applied: kernel=%dx%d open_iter=%d close_iter=%d nonzero_before=%d nonzero_after=%d",
        k, k, cfg.morph_open_iterations, cfg.morph_close_iterations,
        int(np.count_nonzero(mask)), int(np.count_nonzero(closed)),
    )
    return closed


def dilate_mask(mask: np.ndarray, iterations: int = 1, kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)
