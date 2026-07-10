"""
classify.py
===========
Stage 8: Classifies every match/unmatched region produced by Stage 7 into
a final change category:
    ADDED | REMOVED | MODIFIED | GEOMETRY_CHANGE | DIMENSION_CHANGE | TEXT_CHANGE

A spatial rule engine breaks ties when multiple signals disagree (e.g. a
region with both a geometry change AND a text change is reported as
MODIFIED with sub-reasons attached), and assigns a final confidence score
used for visualization and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from config.config import settings
from modules.matching.matcher import MatchedPair, MatchingResult
from modules.ocr.parser import text_similarity
from modules.utils.geometry import BoundingBox, DetectedRegion
from modules.utils.logger import get_logger

logger = get_logger(__name__)


class ChangeCategory(str, Enum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    MODIFIED = "MODIFIED"
    GEOMETRY_CHANGE = "GEOMETRY_CHANGE"
    DIMENSION_CHANGE = "DIMENSION_CHANGE"
    TEXT_CHANGE = "TEXT_CHANGE"
    UNCHANGED = "UNCHANGED"


@dataclass
class ClassifiedChange:
    id: str
    category: ChangeCategory
    bbox: BoundingBox
    confidence: float
    reasons: List[str] = field(default_factory=list)
    before_text: str = ""
    after_text: str = ""
    before_value: Optional[float] = None
    after_value: Optional[float] = None
    object_class: str = "unknown"
    source: str = "cv"


def _confidence_bucket(score: float) -> str:
    cfg = settings.classification
    if score >= cfg.confidence_high:
        return "high"
    if score >= cfg.confidence_medium:
        return "medium"
    return "low"


def _classify_pair(pair: MatchedPair) -> Optional[ClassifiedChange]:
    """Classify a matched before/after pair into MODIFIED sub-categories, or None if unchanged."""
    cfg = settings.classification
    before, after = pair.before, pair.after
    reasons: List[str] = []

    iou = before.bbox.iou(after.bbox)
    geometry_changed = iou < cfg.geometry_iou_threshold

    dim_changed = False
    if before.numeric_value is not None and after.numeric_value is not None:
        denom = max(abs(before.numeric_value), abs(after.numeric_value), 1e-6)
        diff_ratio = abs(before.numeric_value - after.numeric_value) / denom
        dim_changed = diff_ratio > cfg.dimension_change_tolerance
        if dim_changed:
            reasons.append(
                f"Dimension changed from {before.numeric_value:g} to {after.numeric_value:g}"
            )

    text_changed = False
    if before.ocr_text or after.ocr_text:
        sim = text_similarity(before.ocr_text, after.ocr_text)
        text_changed = sim < cfg.text_similarity_threshold
        if text_changed and not dim_changed:
            reasons.append(f"Text changed from '{before.ocr_text}' to '{after.ocr_text}'")

    if geometry_changed:
        reasons.append(f"Geometry changed (IoU={iou:.2f})")

    if not (geometry_changed or dim_changed or text_changed):
        return None  # effectively unchanged; not reported

    # Priority: dimension change is most specific/actionable, then text, then pure geometry.
    # If more than one signal fires, report MODIFIED as the umbrella category.
    signals_fired = sum([geometry_changed, dim_changed, text_changed])
    if signals_fired > 1:
        category = ChangeCategory.MODIFIED
    elif dim_changed:
        category = ChangeCategory.DIMENSION_CHANGE
    elif text_changed:
        category = ChangeCategory.TEXT_CHANGE
    else:
        category = ChangeCategory.GEOMETRY_CHANGE

    confidence = float(pair.match_score * (0.6 + 0.4 * (signals_fired / 3.0)))

    return ClassifiedChange(
        id=after.id,
        category=category,
        bbox=after.bbox,
        confidence=round(min(1.0, confidence), 4),
        reasons=reasons,
        before_text=before.ocr_text,
        after_text=after.ocr_text,
        before_value=before.numeric_value,
        after_value=after.numeric_value,
        object_class=after.object_class if after.object_class != "unknown" else before.object_class,
        source=after.source,
    )


def classify_changes(matching_result: MatchingResult) -> List[ClassifiedChange]:
    """
    Run Stage 8 classification over an entire matching result, producing
    the final list of reportable changes (added, removed, and modified
    sub-categories). Truly unchanged matched pairs are dropped.
    """
    changes: List[ClassifiedChange] = []

    for pair in matching_result.matched_pairs:
        classified = _classify_pair(pair)
        if classified is not None:
            changes.append(classified)

    for region in matching_result.removed:
        changes.append(
            ClassifiedChange(
                id=region.id,
                category=ChangeCategory.REMOVED,
                bbox=region.bbox,
                confidence=round(max(0.5, region.detector_confidence or 0.75), 4),
                reasons=["Present in BEFORE, absent in AFTER"],
                before_text=region.ocr_text,
                object_class=region.object_class,
                source=region.source,
            )
        )

    for region in matching_result.added:
        changes.append(
            ClassifiedChange(
                id=region.id,
                category=ChangeCategory.ADDED,
                bbox=region.bbox,
                confidence=round(max(0.5, region.detector_confidence or 0.75), 4),
                reasons=["Absent in BEFORE, present in AFTER"],
                after_text=region.ocr_text,
                object_class=region.object_class,
                source=region.source,
            )
        )

    logger.info("Classification complete: %d reportable changes.", len(changes))
    return changes
