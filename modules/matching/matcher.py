"""
matcher.py
==========
Stage 7b: Builds the multi-factor cost matrix between BEFORE and AFTER
candidate regions and solves it with the Hungarian algorithm. The match
score combines: spatial distance, object type agreement, dimension value
closeness, direction consistency, geometry (contour) similarity, OCR text
similarity, and IoU — each independently weighted via config.py.

Regions from BEFORE with no valid match => REMOVED.
Regions from AFTER with no valid match => ADDED.
Matched pairs are passed to the classification stage to decide whether
they're identical, geometry-changed, dimension-changed, or text-changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config.config import settings
from modules.matching.hungarian import solve_assignment
from modules.ocr.parser import text_similarity
from modules.utils.geometry import DetectedRegion
from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchedPair:
    before: DetectedRegion
    after: DetectedRegion
    match_score: float  # 1.0 = perfect match, 0.0 = no similarity


@dataclass
class MatchingResult:
    matched_pairs: List[MatchedPair]
    removed: List[DetectedRegion]  # unmatched BEFORE regions
    added: List[DetectedRegion]    # unmatched AFTER regions


def _type_similarity(a: DetectedRegion, b: DetectedRegion) -> float:
    if a.object_class == "unknown" or b.object_class == "unknown":
        return 0.5  # neutral when semantic class isn't available (YOLO inactive)
    return 1.0 if a.object_class == b.object_class else 0.0


def _dimension_similarity(a: DetectedRegion, b: DetectedRegion) -> float:
    if a.numeric_value is None or b.numeric_value is None:
        return 0.5
    denom = max(abs(a.numeric_value), abs(b.numeric_value), 1e-6)
    diff_ratio = abs(a.numeric_value - b.numeric_value) / denom
    return float(max(0.0, 1.0 - diff_ratio))


def _direction_similarity(a: DetectedRegion, b: DetectedRegion, global_shift: Optional[tuple]) -> float:
    """Rewards pairs whose relative displacement matches the page's dominant shift vector."""
    if global_shift is None:
        return 0.5
    dx, dy = a.bbox.direction_vector_to(b.bbox)
    gx, gy = global_shift
    mag_a = (dx ** 2 + dy ** 2) ** 0.5
    mag_g = (gx ** 2 + gy ** 2) ** 0.5
    if mag_a < 1e-6 or mag_g < 1e-6:
        return 1.0 if mag_a < 1e-6 else 0.5
    cos_sim = (dx * gx + dy * gy) / (mag_a * mag_g)
    return float(max(0.0, (cos_sim + 1.0) / 2.0))


def _geometry_similarity(a: DetectedRegion, b: DetectedRegion) -> float:
    """Compares width/height aspect ratio and size as a lightweight geometry proxy."""
    aw, ah = a.bbox.width, a.bbox.height
    bw, bh = b.bbox.width, b.bbox.height
    if aw == 0 or ah == 0 or bw == 0 or bh == 0:
        return 0.0
    size_ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh)
    aspect_a = aw / ah
    aspect_b = bw / bh
    aspect_ratio = min(aspect_a, aspect_b) / max(aspect_a, aspect_b)
    return float(0.5 * size_ratio + 0.5 * aspect_ratio)


def _pair_score(a: DetectedRegion, b: DetectedRegion, global_shift: Optional[tuple]) -> float:
    cfg = settings.matching
    distance = a.bbox.center_distance(b.bbox)

    # Hard spatial cutoff: regions further apart than max_match_distance_px
    # cannot represent the same physical element, regardless of how
    # "neutral" the other similarity signals look (prevents spurious
    # matches when only one candidate exists on each side).
    if distance > cfg.max_match_distance_px:
        return 0.0

    distance_score = max(0.0, 1.0 - min(1.0, distance / cfg.max_match_distance_px))

    type_score = _type_similarity(a, b)
    dimension_score = _dimension_similarity(a, b)
    direction_score = _direction_similarity(a, b, global_shift)
    geometry_score = _geometry_similarity(a, b)
    text_score = text_similarity(a.ocr_text, b.ocr_text) if (a.ocr_text or b.ocr_text) else 0.5
    iou_score = a.bbox.iou(b.bbox)

    total = (
        cfg.weight_distance * distance_score
        + cfg.weight_type * type_score
        + cfg.weight_dimension * dimension_score
        + cfg.weight_direction * direction_score
        + cfg.weight_geometry * geometry_score
        + cfg.weight_text * text_score
        + cfg.weight_iou * iou_score
    )
    return float(np.clip(total, 0.0, 1.0))


def estimate_global_shift(before_regions: List[DetectedRegion], after_regions: List[DetectedRegion]) -> Optional[tuple]:
    """Estimate the dominant translation vector between region sets via centroid-of-centroids."""
    if not before_regions or not after_regions:
        return None
    before_centers = np.array([r.bbox.center for r in before_regions])
    after_centers = np.array([r.bbox.center for r in after_regions])
    shift = after_centers.mean(axis=0) - before_centers.mean(axis=0)
    return (float(shift[0]), float(shift[1]))


def match_regions(
    before_regions: List[DetectedRegion],
    after_regions: List[DetectedRegion],
) -> MatchingResult:
    """
    Match BEFORE and AFTER candidate regions using a weighted multi-factor
    cost matrix solved with the Hungarian algorithm.
    """
    cfg = settings.matching

    if not before_regions and not after_regions:
        return MatchingResult([], [], [])
    if not before_regions:
        return MatchingResult([], [], list(after_regions))
    if not after_regions:
        return MatchingResult([], list(before_regions), [])

    global_shift = estimate_global_shift(before_regions, after_regions)

    n_before, n_after = len(before_regions), len(after_regions)
    score_matrix = np.zeros((n_before, n_after), dtype=np.float64)

    for i, b in enumerate(before_regions):
        for j, a in enumerate(after_regions):
            score_matrix[i, j] = _pair_score(b, a, global_shift)

    # Hungarian minimizes cost, so convert similarity -> cost
    cost_matrix = 1.0 - score_matrix
    assignment = solve_assignment(cost_matrix, max_cost=1.0 - cfg.match_score_threshold)

    matched_pairs: List[MatchedPair] = []
    matched_before_idx = set()
    matched_after_idx = set()

    for bi, ai, cost in zip(assignment.before_indices, assignment.after_indices, assignment.costs):
        matched_pairs.append(
            MatchedPair(before=before_regions[bi], after=after_regions[ai], match_score=1.0 - cost)
        )
        matched_before_idx.add(bi)
        matched_after_idx.add(ai)

    removed = [r for idx, r in enumerate(before_regions) if idx not in matched_before_idx]
    added = [r for idx, r in enumerate(after_regions) if idx not in matched_after_idx]

    logger.info(
        "Matching complete: %d matched pairs, %d removed, %d added (before=%d, after=%d)",
        len(matched_pairs), len(removed), len(added), n_before, n_after,
    )

    return MatchingResult(matched_pairs=matched_pairs, removed=removed, added=added)
