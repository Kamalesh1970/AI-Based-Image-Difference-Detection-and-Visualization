"""
ssim.py
=======
Structural Similarity Index (SSIM) based difference map generation.
This is ONE signal among several combined in the hybrid detection pipeline
(see detection/components.py for the fusion stage) -- per the project
requirements, SSIM alone is never used as the sole detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SSIMResult:
    score: float
    diff_map: np.ndarray       # float64 [0,1], 1 = identical
    binary_diff_mask: np.ndarray  # uint8 {0, 255}, 255 = changed


def compute_ssim_diff(before: np.ndarray, after: np.ndarray) -> SSIMResult:
    """
    Compute the SSIM similarity map between two aligned grayscale/BGR images.

    Returns:
        SSIMResult with the global score, the raw diff map, and a
        thresholded binary change mask.
    """
    cfg = settings.detection
    before_gray = before if len(before.shape) == 2 else cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = after if len(after.shape) == 2 else cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    if before_gray.shape != after_gray.shape:
        after_gray = cv2.resize(after_gray, (before_gray.shape[1], before_gray.shape[0]))

    win_size = cfg.ssim_win_size
    if win_size % 2 == 0:
        win_size += 1
    win_size = min(win_size, min(before_gray.shape) - 1 if min(before_gray.shape) % 2 == 0 else min(before_gray.shape))
    win_size = max(3, win_size)

    score, diff = ssim(before_gray, after_gray, full=True, win_size=win_size)
    diff_map = diff.astype(np.float64)

    # Regions where structural similarity drops below the threshold are "changed"
    change_mask = (diff_map < cfg.ssim_diff_threshold).astype(np.uint8) * 255

    logger.info("SSIM global score=%.4f, threshold=%.2f", score, cfg.ssim_diff_threshold)

    return SSIMResult(score=float(score), diff_map=diff_map, binary_diff_mask=change_mask)
