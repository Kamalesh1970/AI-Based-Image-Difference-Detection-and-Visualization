"""
edges.py
========
Canny-edge based difference signal. Captures fine line-work changes
(new/removed edges, moved geometry boundaries) that SSIM's windowed
statistics can miss, especially for thin CAD lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EdgeDiffResult:
    before_edges: np.ndarray
    after_edges: np.ndarray
    edge_diff_mask: np.ndarray  # uint8 {0,255}


def compute_edge_diff(before: np.ndarray, after: np.ndarray) -> EdgeDiffResult:
    """
    Run Canny edge detection on both images and compute a symmetric
    difference mask (edges present in one image but not the other).
    """
    cfg = settings.detection
    before_gray = before if len(before.shape) == 2 else cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = after if len(after.shape) == 2 else cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    if before_gray.shape != after_gray.shape:
        after_gray = cv2.resize(after_gray, (before_gray.shape[1], before_gray.shape[0]))

    before_edges = cv2.Canny(before_gray, cfg.canny_low, cfg.canny_high)
    after_edges = cv2.Canny(after_gray, cfg.canny_low, cfg.canny_high)

    # Dilate slightly before XOR to tolerate 1-2px sub-pixel alignment jitter
    kernel = np.ones((3, 3), np.uint8)
    before_dilated = cv2.dilate(before_edges, kernel, iterations=1)
    after_dilated = cv2.dilate(after_edges, kernel, iterations=1)

    added_edges = cv2.bitwise_and(after_dilated, cv2.bitwise_not(before_dilated))
    removed_edges = cv2.bitwise_and(before_dilated, cv2.bitwise_not(after_dilated))
    edge_diff_mask = cv2.bitwise_or(added_edges, removed_edges)

    logger.debug(
        "Edge diff computed: before_edge_px=%d after_edge_px=%d diff_px=%d",
        int(np.count_nonzero(before_edges)),
        int(np.count_nonzero(after_edges)),
        int(np.count_nonzero(edge_diff_mask)),
    )

    return EdgeDiffResult(
        before_edges=before_edges,
        after_edges=after_edges,
        edge_diff_mask=edge_diff_mask,
    )
