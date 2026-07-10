"""
components.py
==============
Stage 4 fusion: combines SSIM + Canny edge difference signals into one
fused change mask, cleans it with morphology, then extracts connected
components / contours and filters them by area and perimeter to produce
the final list of candidate change regions for this image pair.

This is the module the project spec refers to as combining
"SSIM + Canny Edge + Morphological Operations + Connected Components +
Contour Filtering".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from config.config import settings
from modules.detection.edges import compute_edge_diff
from modules.detection.morphology import clean_mask
from modules.detection.ssim import compute_ssim_diff
from modules.utils.geometry import BoundingBox
from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateRegion:
    bbox: BoundingBox
    area: int
    perimeter: float
    fused_score: float  # 0..1 confidence that this is a genuine change


@dataclass
class FusionResult:
    fused_mask: np.ndarray
    ssim_score: float
    candidates: List[CandidateRegion]


def _fuse_masks(ssim_mask: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """OR-fuse the SSIM structural mask with the edge-difference mask."""
    if ssim_mask.shape != edge_mask.shape:
        edge_mask = cv2.resize(edge_mask, (ssim_mask.shape[1], ssim_mask.shape[0]))
    return cv2.bitwise_or(ssim_mask, edge_mask)


def detect_candidate_regions(before: np.ndarray, after: np.ndarray) -> FusionResult:
    """
    Run the full Stage-4 fusion pipeline on an aligned before/after image pair.

    Returns:
        FusionResult with the fused binary mask, global SSIM score, and a
        filtered list of CandidateRegion objects (bounding box + confidence).
    """
    cfg = settings.detection

    ssim_result = compute_ssim_diff(before, after)
    edge_result = compute_edge_diff(before, after)

    fused = _fuse_masks(ssim_result.binary_diff_mask, edge_result.edge_diff_mask)
    cleaned = clean_mask(fused)

    # Connected components (4-way) for initial candidate extraction
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    h, w = cleaned.shape[:2]
    max_area = int(cfg.max_component_area_ratio * h * w)

    candidates: List[CandidateRegion] = []
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < cfg.min_component_area:
            continue
        if area > max_area:
            # Likely a global alignment artifact or page-wide shift; skip.
            continue
        if perimeter < cfg.contour_min_perimeter:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = BoundingBox(x, y, x + bw, y + bh).pad(
            cfg.box_padding_px, max_w=w, max_h=h
        )

        # Confidence heuristic: combine local SSIM dissimilarity within the
        # box with edge density, both normalized to [0, 1].
        local_ssim_patch = ssim_result.diff_map[y: y + bh, x: x + bw]
        local_dissimilarity = float(1.0 - np.mean(local_ssim_patch)) if local_ssim_patch.size else 0.0
        local_edge_density = float(
            np.count_nonzero(edge_result.edge_diff_mask[y: y + bh, x: x + bw])
        ) / max(1, bw * bh)

        fused_score = float(np.clip(0.6 * local_dissimilarity + 0.4 * min(1.0, local_edge_density * 5), 0.0, 1.0))

        candidates.append(
            CandidateRegion(bbox=bbox, area=int(area), perimeter=float(perimeter), fused_score=fused_score)
        )

    logger.info(
        "Stage-4 fusion: %d raw contours -> %d filtered candidates (ssim_score=%.4f)",
        len(contours), len(candidates), ssim_result.score,
    )

    return FusionResult(fused_mask=cleaned, ssim_score=ssim_result.score, candidates=candidates)
