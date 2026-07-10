"""
legend.py
=========
Renders a color-coded legend panel and appends it to the annotated
comparison image, plus a compact statistics summary (counts per category),
so the final output image is self-explanatory without external docs.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import cv2
import numpy as np

from config.config import settings
from modules.classification.classify import ChangeCategory, ClassifiedChange
from modules.utils.logger import get_logger

logger = get_logger(__name__)

_LEGEND_ENTRIES = [
    ("Added", "color_added"),
    ("Removed", "color_removed"),
    ("Modified / Dimension / Text", "color_modified"),
    ("Geometry Changed", "color_geometry"),
]


def build_legend_panel(height: int, changes: List[ClassifiedChange]) -> np.ndarray:
    """Build a standalone legend + stats panel image of the given height."""
    cfg = settings.visualization
    width = cfg.legend_width_px
    panel = np.full((height, width, 3), 250, dtype=np.uint8)

    y = 30
    cv2.putText(panel, "LEGEND", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    y += 30

    for label, color_attr in _LEGEND_ENTRIES:
        color = getattr(cfg, color_attr)
        cv2.rectangle(panel, (14, y - 14), (34, y + 4), color, -1)
        cv2.putText(panel, label, (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        y += 32

    y += 10
    cv2.line(panel, (14, y), (width - 14, y), (200, 200, 200), 1)
    y += 30

    cv2.putText(panel, "STATISTICS", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    y += 28

    counts = Counter(c.category.value for c in changes)
    cv2.putText(panel, f"Total changes: {len(changes)}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
    y += 24
    for category in ChangeCategory:
        if category == ChangeCategory.UNCHANGED:
            continue
        count = counts.get(category.value, 0)
        if count == 0:
            continue
        cv2.putText(
            panel, f"{category.value}: {count}", (14, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA,
        )
        y += 22

    return panel


def append_legend(annotated_image: np.ndarray, changes: List[ClassifiedChange]) -> np.ndarray:
    """Horizontally concatenate the legend/stats panel onto the annotated image."""
    h = annotated_image.shape[0]
    legend = build_legend_panel(h, changes)
    combined = np.hstack([annotated_image, legend])
    logger.debug("Legend panel appended (%dx%d).", legend.shape[1], legend.shape[0])
    return combined
