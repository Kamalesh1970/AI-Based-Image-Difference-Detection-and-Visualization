"""
preprocess.py (ocr)
====================
Stage 5a: Region-of-interest preprocessing before OCR. Small text regions
cropped from a CAD drawing need aggressive upscaling and sharpening for
Tesseract to reliably recognize dimension values and notes.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_roi_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Upscale, sharpen, and binarize a small ROI crop to maximize OCR
    accuracy for dense engineering text/dimensions.

    Args:
        roi: BGR or grayscale crop of the region containing text.

    Returns:
        A clean, upscaled binary image ready for pytesseract.
    """
    if roi is None or roi.size == 0:
        raise ValueError("prepare_roi_for_ocr received an empty ROI.")

    cfg = settings.ocr
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Upscale using cubic interpolation for smoother edges on tiny glyphs
    factor = max(1.0, cfg.upscale_factor)
    upscaled = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    # Unsharp-mask style sharpening
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        upscaled, 1 + cfg.sharpen_strength, blurred, -cfg.sharpen_strength, 0
    )

    # Otsu threshold after sharpening for a crisp binary result
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure black text on white background (Tesseract's preferred polarity)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Slight dilation to reconnect thin/broken character strokes after upscaling
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    logger.debug("OCR ROI preprocessed: original=%s upscaled=%s", roi.shape[:2], binary.shape[:2])
    return binary


def add_padding(roi: np.ndarray, padding: int = 10) -> np.ndarray:
    """Add white padding around a binary ROI; helps Tesseract's layout analysis."""
    return cv2.copyMakeBorder(
        roi, padding, padding, padding, padding,
        borderType=cv2.BORDER_CONSTANT, value=255,
    )
