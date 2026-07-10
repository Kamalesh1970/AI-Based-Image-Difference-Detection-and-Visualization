"""
clustering.py
=============
Spatial clustering of candidate change regions using DBSCAN on box
centers. This merges fragmented detections that belong to the same
logical change (e.g. a dimension line broken into several contour pieces
by morphological gaps) into a single unified region, and removes isolated
noisy micro-regions with no neighbors.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from config.config import settings
from modules.detection.components import CandidateRegion
from modules.utils.geometry import BoundingBox
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def cluster_candidates(candidates: List[CandidateRegion]) -> List[CandidateRegion]:
    """
    Cluster candidate regions by spatial proximity (DBSCAN on box centers)
    and merge each cluster into a single bounding candidate region whose
    confidence is the max of its members.

    Args:
        candidates: raw candidate regions from the fusion stage.

    Returns:
        A reduced list of merged CandidateRegion objects.
    """
    if not candidates:
        return []

    cfg = settings.detection
    centers = np.array([c.bbox.center for c in candidates])

    clustering = DBSCAN(
        eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples, metric="euclidean"
    ).fit(centers)

    labels = clustering.labels_
    merged: List[CandidateRegion] = []

    for label in set(labels):
        member_indices = np.where(labels == label)[0]
        members = [candidates[i] for i in member_indices]

        merged_bbox = members[0].bbox
        total_area = 0
        max_perimeter = 0.0
        max_score = 0.0
        for m in members[1:]:
            merged_bbox = merged_bbox.union(m.bbox)
        for m in members:
            total_area += m.area
            max_perimeter = max(max_perimeter, m.perimeter)
            max_score = max(max_score, m.fused_score)

        merged.append(
            CandidateRegion(
                bbox=merged_bbox,
                area=total_area,
                perimeter=max_perimeter,
                fused_score=max_score,
            )
        )

    logger.info("DBSCAN clustering: %d raw candidates -> %d merged regions", len(candidates), len(merged))
    return merged


def merge_close_boxes(boxes: List[BoundingBox], max_distance_px: int | None = None) -> List[BoundingBox]:
    """
    Simple greedy proximity merge (center-distance based), used as a
    lighter-weight alternative to DBSCAN in the visualization stage when
    finalizing box layout.
    """
    cfg = settings.detection
    max_distance_px = max_distance_px or cfg.cluster_merge_distance_px
    if not boxes:
        return []

    remaining = list(boxes)
    merged: List[BoundingBox] = []

    while remaining:
        base = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            still = []
            for b in remaining:
                if base.center_distance(b) <= max_distance_px:
                    base = base.union(b)
                    changed = True
                else:
                    still.append(b)
            remaining = still
        merged.append(base)

    return merged
