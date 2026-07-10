"""
deskew.py
=========
Stage 2b: Automatic skew detection & correction for rendered drawing pages.
Uses a projection-profile variance method (robust for CAD drawings with
strong horizontal/vertical line content) with a Hough-transform fallback.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def _projection_score(gray: np.ndarray, angle: float) -> float:
    """Rotate and compute the variance of the horizontal projection profile."""
    h, w = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )
    profile = np.sum(rotated, axis=1).astype(np.float64)
    return float(np.var(profile))


def estimate_skew_angle(image: np.ndarray) -> float:
    """
    Estimate the skew angle (degrees) of a drawing using a coarse-to-fine
    projection-profile search bounded by config.deskew_angle_limit.
    """
    cfg = settings.preprocess
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize for a cleaner projection signal
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -1.0
    coarse_range = np.arange(-cfg.deskew_angle_limit, cfg.deskew_angle_limit + 1e-9, 0.5)
    for angle in coarse_range:
        score = _projection_score(binary, float(angle))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    # Fine search around the coarse best angle
    fine_range = np.arange(
        best_angle - 0.5, best_angle + 0.5 + 1e-9, cfg.deskew_angle_step
    )
    for angle in fine_range:
        score = _projection_score(binary, float(angle))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    logger.debug("Estimated skew angle: %.3f degrees", best_angle)
    return best_angle


def deskew_image(image: np.ndarray, angle: float | None = None) -> Tuple[np.ndarray, float]:
    """
    Deskew an image. If `angle` is not provided, it is estimated automatically.

    Returns:
        (deskewed_image, angle_used_degrees)
    """
    if angle is None:
        angle = estimate_skew_angle(image)

    if abs(angle) < 1e-3:
        return image, 0.0

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_value = 255 if len(image.shape) == 2 else (255, 255, 255)
    deskewed = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    logger.info("Deskewed image by %.3f degrees", angle)
    return deskewed, angle
