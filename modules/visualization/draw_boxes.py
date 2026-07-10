"""
draw_boxes.py
=============
Stage 9: Renders numbered, color-coded bounding boxes with confidence
scores onto the AFTER image to produce the final annotated comparison
image. Overlapping boxes of the same category are merged for a clean
visual result before rendering.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from config.config import settings
from modules.classification.classify import ChangeCategory, ClassifiedChange
from modules.utils.logger import get_logger

logger = get_logger(__name__)

_CATEGORY_COLOR_MAP: Dict[ChangeCategory, str] = {
    ChangeCategory.ADDED: "color_added",
    ChangeCategory.REMOVED: "color_removed",
    ChangeCategory.MODIFIED: "color_modified",
    ChangeCategory.DIMENSION_CHANGE: "color_modified",
    ChangeCategory.TEXT_CHANGE: "color_modified",
    ChangeCategory.GEOMETRY_CHANGE: "color_geometry",
}


def _get_color(category: ChangeCategory) -> tuple:
    cfg = settings.visualization
    attr = _CATEGORY_COLOR_MAP.get(category, "color_geometry")
    return getattr(cfg, attr)


def _draw_label(
    image: np.ndarray, text: str, origin: tuple, color: tuple, cfg
) -> None:
    """Draw a filled semi-transparent label background with text on top."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, cfg.font_scale, cfg.font_thickness)
    x, y = origin
    overlay = image.copy()
    cv2.rectangle(
        overlay, (x, y - text_h - baseline - 4), (x + text_w + 6, y + 2), color, -1
    )
    cv2.addWeighted(overlay, cfg.label_bg_alpha, image, 1 - cfg.label_bg_alpha, 0, image)
    cv2.putText(
        image, text, (x + 3, y - 4), font, cfg.font_scale, (255, 255, 255), cfg.font_thickness, cv2.LINE_AA
    )


def draw_annotated_image(base_image: np.ndarray, changes: List[ClassifiedChange]) -> np.ndarray:
    """
    Draw numbered, color-coded, confidence-annotated bounding boxes for
    every classified change onto a copy of `base_image`.

    Args:
        base_image: typically the AFTER (aligned) image.
        changes: classified changes from Stage 8.

    Returns:
        A new BGR image with all annotations drawn.
    """
    cfg = settings.visualization
    output = base_image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for idx, change in enumerate(changes, start=1):
        color = _get_color(change.category)
        box = change.bbox

        cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), color, cfg.box_thickness)

        # Numbered ID circle at top-left corner of the box
        circle_center = (box.x1, box.y1)
        cv2.circle(output, circle_center, cfg.id_circle_radius, color, -1)
        cv2.putText(
            output, str(idx),
            (circle_center[0] - 6 if idx < 10 else circle_center[0] - 10, circle_center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, (255, 255, 255), cfg.font_thickness, cv2.LINE_AA,
        )

        label = f"#{idx} {change.category.value} ({change.confidence * 100:.0f}%)"
        label_origin = (box.x1, max(20, box.y1 - 6))
        _draw_label(output, label, label_origin, color, cfg)

    logger.info("Rendered annotated image with %d change boxes.", len(changes))
    return output


def merge_same_category_overlaps(changes: List[ClassifiedChange]) -> List[ClassifiedChange]:
    """
    Merge boxes of the SAME category whose IoU exceeds the configured
    overlap threshold, to avoid visually cluttered duplicate boxes.
    Keeps the higher-confidence change's metadata for the merged box.
    """
    cfg = settings.visualization
    if not changes:
        return []

    remaining = list(changes)
    merged: List[ClassifiedChange] = []

    while remaining:
        base = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            still = []
            for c in remaining:
                if c.category == base.category and base.bbox.iou(c.bbox) >= cfg.overlap_merge_iou:
                    if c.confidence > base.confidence:
                        base.confidence = c.confidence
                        base.reasons = list(set(base.reasons + c.reasons))
                    base.bbox = base.bbox.union(c.bbox)
                    changed = True
                else:
                    still.append(c)
            remaining = still
        merged.append(base)

    logger.debug("Merged %d changes into %d after overlap consolidation.", len(changes), len(merged))
    return merged
