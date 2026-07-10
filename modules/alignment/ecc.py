"""
ecc.py
======
Stage 3: Image alignment using the Enhanced Correlation Coefficient (ECC)
algorithm (Evangelidis & Psarakis, 2008). ECC is chosen deliberately over
feature-based methods (ORB / SIFT / AKAZE) because engineering drawings are
mostly thin line-work with sparse, repetitive, and often symmetric features
that cause feature matchers to mismatch. ECC directly optimizes pixel-wise
photometric alignment and achieves reliable sub-pixel precision on this kind
of content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)

_MOTION_MAP = {
    "TRANSLATION": cv2.MOTION_TRANSLATION,
    "EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
    "AFFINE": cv2.MOTION_AFFINE,
    "HOMOGRAPHY": cv2.MOTION_HOMOGRAPHY,
}


class ECCAlignmentError(RuntimeError):
    """Raised when ECC fails to converge and no usable transform is found."""


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    warp_matrix: np.ndarray
    motion_type: str
    correlation_coefficient: float
    converged: bool


def _prepare_gray(image: np.ndarray, blur_size: int) -> np.ndarray:
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    if blur_size > 0:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def align_ecc(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    motion_type: str | None = None,
) -> AlignmentResult:
    """
    Align `moving_image` onto `reference_image` using ECC, run on an
    image pyramid for robustness against larger initial misalignment,
    then refined at full resolution for sub-pixel accuracy.

    Args:
        reference_image: the BEFORE image (target coordinate frame).
        moving_image: the AFTER image, to be warped onto the reference frame.
        motion_type: one of TRANSLATION / EUCLIDEAN / AFFINE / HOMOGRAPHY.

    Returns:
        AlignmentResult containing the warped `moving_image` and the warp matrix.
    """
    cfg = settings.alignment
    motion_type = motion_type or cfg.motion_type
    cv_motion = _MOTION_MAP.get(motion_type.upper(), cv2.MOTION_HOMOGRAPHY)

    ref_h, ref_w = reference_image.shape[:2]
    if moving_image.shape[:2] != (ref_h, ref_w):
        moving_image = cv2.resize(moving_image, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC)

    warp_matrix = (
        np.eye(3, 3, dtype=np.float32)
        if cv_motion == cv2.MOTION_HOMOGRAPHY
        else np.eye(2, 3, dtype=np.float32)
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        cfg.number_of_iterations,
        cfg.termination_eps,
    )

    converged = True
    correlation = -1.0

    try:
        # --- Coarse pass on a downscaled image pyramid for robustness ---
        scale = cfg.downscale_for_ecc
        if 0 < scale < 1.0:
            small_ref = cv2.resize(reference_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small_mov = cv2.resize(moving_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ref_gray_small = _prepare_gray(small_ref, cfg.gaussian_blur_size)
            mov_gray_small = _prepare_gray(small_mov, cfg.gaussian_blur_size)

            coarse_warp = (
                np.eye(3, 3, dtype=np.float32)
                if cv_motion == cv2.MOTION_HOMOGRAPHY
                else np.eye(2, 3, dtype=np.float32)
            )
            try:
                correlation, coarse_warp = cv2.findTransformECC(
                    ref_gray_small, mov_gray_small, coarse_warp, cv_motion, criteria, None, 5,
                )
                # Rescale the translation components back to full resolution
                if cv_motion == cv2.MOTION_HOMOGRAPHY:
                    scale_mat = np.array(
                        [[1, 1, 1 / scale], [1, 1, 1 / scale], [scale, scale, 1]], dtype=np.float32,
                    )
                    warp_matrix = coarse_warp * scale_mat
                else:
                    warp_matrix = coarse_warp.copy()
                    warp_matrix[0, 2] /= scale
                    warp_matrix[1, 2] /= scale
            except cv2.error as exc:
                logger.warning("Coarse ECC pass did not converge, starting from identity: %s", exc)
                converged = False

        # --- Fine pass at full resolution for sub-pixel accuracy ---
        ref_gray = _prepare_gray(reference_image, cfg.gaussian_blur_size)
        mov_gray = _prepare_gray(moving_image, cfg.gaussian_blur_size)

        correlation, warp_matrix = cv2.findTransformECC(
            ref_gray, mov_gray, warp_matrix, cv_motion, criteria, None, 5,
        )
        converged = True

    except cv2.error as exc:
        logger.error("ECC alignment failed to converge: %s. Falling back to identity transform.", exc)
        warp_matrix = (
            np.eye(3, 3, dtype=np.float32)
            if cv_motion == cv2.MOTION_HOMOGRAPHY
            else np.eye(2, 3, dtype=np.float32)
        )
        converged = False
        correlation = 0.0

    if cv_motion == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(
            moving_image, warp_matrix, (ref_w, ref_h),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )
    else:
        aligned = cv2.warpAffine(
            moving_image, warp_matrix, (ref_w, ref_h),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )

    logger.info(
        "ECC alignment complete: motion=%s, correlation=%.4f, converged=%s",
        motion_type, correlation, converged,
    )

    return AlignmentResult(
        aligned_image=aligned,
        warp_matrix=warp_matrix,
        motion_type=motion_type,
        correlation_coefficient=float(correlation),
        converged=converged,
    )
