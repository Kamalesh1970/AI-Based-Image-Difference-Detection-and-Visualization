"""Unit tests for modules.utils.geometry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.utils.geometry import BoundingBox, merge_overlapping_boxes


def test_bbox_basic_properties():
    box = BoundingBox(10, 10, 50, 40)
    assert box.width == 40
    assert box.height == 30
    assert box.area == 1200
    assert box.center == (30.0, 25.0)


def test_bbox_auto_normalizes_reversed_coords():
    box = BoundingBox(50, 40, 10, 10)
    assert box.x1 == 10 and box.x2 == 50
    assert box.y1 == 10 and box.y2 == 40


def test_iou_identical_boxes():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(0, 0, 10, 10)
    assert a.iou(b) == 1.0


def test_iou_disjoint_boxes():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(100, 100, 110, 110)
    assert a.iou(b) == 0.0


def test_iou_partial_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    iou = a.iou(b)
    assert 0.0 < iou < 1.0


def test_union_produces_enclosing_box():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 20, 20)
    u = a.union(b)
    assert u.x1 == 0 and u.y1 == 0 and u.x2 == 20 and u.y2 == 20


def test_pad_respects_bounds():
    box = BoundingBox(5, 5, 10, 10)
    padded = box.pad(3, max_w=12, max_h=12)
    assert padded.x1 == 2 and padded.y1 == 2
    assert padded.x2 == 12 and padded.y2 == 12  # clamped


def test_merge_overlapping_boxes_reduces_count():
    boxes = [
        BoundingBox(0, 0, 10, 10),
        BoundingBox(5, 5, 15, 15),
        BoundingBox(100, 100, 110, 110),
    ]
    merged = merge_overlapping_boxes(boxes, iou_threshold=0.1)
    assert len(merged) == 2


def test_center_distance():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(0, 0, 10, 10)
    assert a.center_distance(b) == 0.0
