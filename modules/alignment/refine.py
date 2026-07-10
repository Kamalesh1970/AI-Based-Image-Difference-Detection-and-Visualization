"""
refine.py
=========
Stage 3b: Sub-pixel alignment refinement. After the coarse/fine ECC pass,
this module applies OpenCV's phase-correlation based `cv2.phaseCorrelate`
to detect and correct any residual sub-pixel translational drift, and
reports an alignment-quality score used to decide whether the pipeline
should proceed or flag the pair for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RefinementResult:
    refined_image: np.ndarray
    residual_shift_x: float
    residual_shift_y: float
    response: float


def refine_subpixel(reference_image: np.ndarray, aligned_image: np.ndarray) -> RefinementResult:
    """
    Detect and correct residual sub-pixel translation between an already
    coarsely-aligned image and the reference, using phase correlation.
    """
    cfg = settings.alignment
    if not cfg.refine_with_phase_correlation:
        return RefinementResult(aligned_image, 0.0, 0.0, 1.0)

    ref_gray = reference_image if len(reference_image.shape) == 2 else cv2.cvtColor(
        reference_image, cv2.COLOR_BGR2GRAY
    )
    mov_gray = aligned_image if len(aligned_image.shape) == 2 else cv2.cvtColor(
        aligned_image, cv2.COLOR_BGR2GRAY
    )

    ref_f = np.float32(ref_gray)
    mov_f = np.float32(mov_gray)

    window = cv2.createHanningWindow((ref_f.shape[1], ref_f.shape[0]), cv2.CV_32F)

    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(ref_f * window, mov_f * window)
    except cv2.error as exc:
        logger.warning("Phase correlation refinement failed: %s. Skipping refinement.", exc)
        return RefinementResult(aligned_image, 0.0, 0.0, 0.0)

    translation_matrix = np.array(
        [[1, 0, -shift_x], [0, 1, -shift_y]], dtype=np.float32,
    )
    h, w = aligned_image.shape[:2]
    refined = cv2.warpAffine(
        aligned_image, translation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )

    logger.info(
        "Sub-pixel refinement: shift=(%.3f, %.3f) px, response=%.4f",
        shift_x, shift_y, response,
    )

    return RefinementResult(
        refined_image=refined,
        residual_shift_x=float(shift_x),
        residual_shift_y=float(shift_y),
        response=float(response),
    )


def compute_alignment_quality(reference_image: np.ndarray, aligned_image: np.ndarray) -> float:
    """
    Compute a normalized [0, 1] alignment-quality score using normalized
    cross-correlation between the two images. Used by the pipeline to warn
    the user if alignment quality is poor before proceeding to diffing.
    """
    ref_gray = reference_image if len(reference_image.shape) == 2 else cv2.cvtColor(
        reference_image, cv2.COLOR_BGR2GRAY
    )
    mov_gray = aligned_image if len(aligned_image.shape) == 2 else cv2.cvtColor(
        aligned_image, cv2.COLOR_BGR2GRAY
    )
    result = cv2.matchTemplate(ref_gray, mov_gray, cv2.TM_CCOEFF_NORMED)
    score = float(np.max(result))
    return max(0.0, min(1.0, (score + 1.0) / 2.0))
