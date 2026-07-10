"""
geometry.py
===========
Shared geometric primitives (bounding boxes) and helper math used across
detection, matching, classification and visualization modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import uuid


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def pad(self, px: int, max_w: Optional[int] = None, max_h: Optional[int] = None) -> "BoundingBox":
        x1 = max(0, self.x1 - px)
        y1 = max(0, self.y1 - px)
        x2 = self.x2 + px if max_w is None else min(max_w, self.x2 + px)
        y2 = self.y2 + px if max_h is None else min(max_h, self.y2 + px)
        return BoundingBox(x1, y1, x2, y2)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def center_distance(self, other: "BoundingBox") -> float:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def direction_vector_to(self, other: "BoundingBox") -> Tuple[float, float]:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return (cx2 - cx1, cy2 - cy1)


@dataclass
class DetectedRegion:
    """
    A region of interest detected either by classical CV difference detection
    or by the YOLO object detector, on ONE image (before or after).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    bbox: BoundingBox = None  # type: ignore
    source: str = "cv"  # 'cv' | 'yolo'
    object_class: str = "unknown"
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    detector_confidence: float = 0.0
    contour_points: Optional[List[Tuple[int, int]]] = None
    numeric_value: Optional[float] = None
    metadata: dict = field(default_factory=dict)


def merge_overlapping_boxes(boxes: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
    """Greedy merge of boxes whose IoU exceeds the threshold. Returns a new, reduced list."""
    if not boxes:
        return []

    merged: List[BoundingBox] = []
    remaining = list(boxes)

    while remaining:
        base = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            still_remaining = []
            for b in remaining:
                if base.iou(b) >= iou_threshold or _contains(base, b) or _contains(b, base):
                    base = base.union(b)
                    changed = True
                else:
                    still_remaining.append(b)
            remaining = still_remaining
        merged.append(base)

    return merged


def _contains(a: BoundingBox, b: BoundingBox) -> bool:
    """True if box b's center lies inside box a (handles nested small boxes)."""
    cx, cy = b.center
    return a.x1 <= cx <= a.x2 and a.y1 <= cy <= a.y2
