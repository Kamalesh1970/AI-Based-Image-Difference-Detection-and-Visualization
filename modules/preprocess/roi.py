"""
roi.py
======
Stage 2d: Automatic detection and cropping of the main drawing area,
excluding surrounding scanner margins / whitespace. This ensures the
alignment and difference-detection stages operate only on meaningful
content, reducing false positives from irrelevant page borders.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from config.config import settings
from modules.utils.geometry import BoundingBox
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def detect_drawing_roi(image: np.ndarray) -> BoundingBox:
    """
    Detect the bounding box of non-white drawing content.

    Args:
        image: BGR or grayscale image.

    Returns:
        BoundingBox of the detected drawing area (in original image coords).
        Falls back to the full image if no content is found.
    """
    cfg = settings.preprocess
    h, w = image.shape[:2]
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        logger.warning("No drawing content detected; using full image as ROI.")
        return BoundingBox(0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(coords)
    margin_x = int(bw * cfg.roi_margin_ratio)
    margin_y = int(bh * cfg.roi_margin_ratio)

    box = BoundingBox(
        x1=max(0, x - margin_x),
        y1=max(0, y - margin_y),
        x2=min(w, x + bw + margin_x),
        y2=min(h, y + bh + margin_y),
    )
    logger.info("Detected drawing ROI: %s (source image %dx%d)", box.as_tuple(), w, h)
    return box


def crop_to_roi(image: np.ndarray, roi: BoundingBox) -> np.ndarray:
    """Crop an image to the given ROI bounding box."""
    return image[roi.y1: roi.y2, roi.x1: roi.x2]


def auto_crop_drawing_area(image: np.ndarray) -> Tuple[np.ndarray, BoundingBox]:
    """
    Convenience wrapper: detect and crop the drawing area in one call,
    respecting the `auto_crop_drawing_area` config flag.
    """
    if not settings.preprocess.auto_crop_drawing_area:
        h, w = image.shape[:2]
        return image, BoundingBox(0, 0, w, h)

    roi = detect_drawing_roi(image)
    cropped = crop_to_roi(image, roi)
    return cropped, roi
