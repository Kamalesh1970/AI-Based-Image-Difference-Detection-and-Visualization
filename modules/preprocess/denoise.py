"""
denoise.py
==========
Stage 2a: Noise removal for scanned/rendered engineering drawings using
Non-Local Means denoising, which preserves thin line-work better than
Gaussian or median blurring.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means denoising to a grayscale or BGR image.

    Args:
        image: input image (grayscale or BGR).

    Returns:
        Denoised image of the same shape/dtype as the input.
    """
    if image is None or image.size == 0:
        raise ValueError("denoise_image received an empty image.")

    cfg = settings.preprocess
    try:
        if len(image.shape) == 2:
            result = cv2.fastNlMeansDenoising(
                image,
                h=cfg.denoise_h,
                templateWindowSize=cfg.denoise_template_window,
                searchWindowSize=cfg.denoise_search_window,
            )
        else:
            result = cv2.fastNlMeansDenoisingColored(
                image,
                h=cfg.denoise_h,
                hColor=cfg.denoise_h,
                templateWindowSize=cfg.denoise_template_window,
                searchWindowSize=cfg.denoise_search_window,
            )
        logger.debug("Denoising applied: shape=%s", image.shape)
        return result
    except cv2.error as exc:
        logger.error("Denoising failed, returning original image: %s", exc)
        return image


def remove_salt_and_pepper(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median-blur based salt-and-pepper noise removal, useful pre-binarization."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)
