"""Unit tests for modules.matching.matcher using synthetic DetectedRegions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.matching.matcher import match_regions
from modules.utils.geometry import BoundingBox, DetectedRegion


def make_region(x1, y1, x2, y2, text="", value=None, cls="unknown"):
    return DetectedRegion(
        bbox=BoundingBox(x1, y1, x2, y2),
        ocr_text=text,
        numeric_value=value,
        object_class=cls,
    )


def test_match_identical_single_region():
    before = [make_region(10, 10, 50, 50, text="450mm", value=450.0)]
    after = [make_region(10, 10, 50, 50, text="450mm", value=450.0)]
    result = match_regions(before, after)
    assert len(result.matched_pairs) == 1
    assert len(result.removed) == 0
    assert len(result.added) == 0


def test_unmatched_before_is_removed():
    before = [make_region(10, 10, 50, 50)]
    after = []
    result = match_regions(before, after)
    assert len(result.removed) == 1
    assert len(result.matched_pairs) == 0


def test_unmatched_after_is_added():
    before = []
    after = [make_region(10, 10, 50, 50)]
    result = match_regions(before, after)
    assert len(result.added) == 1
    assert len(result.matched_pairs) == 0


def test_far_apart_regions_are_not_matched():
    before = [make_region(0, 0, 10, 10)]
    after = [make_region(2000, 2000, 2010, 2010)]
    result = match_regions(before, after)
    assert len(result.matched_pairs) == 0
    assert len(result.removed) == 1
    assert len(result.added) == 1


def test_empty_inputs():
    result = match_regions([], [])
    assert result.matched_pairs == []
    assert result.removed == []
    assert result.added == []
