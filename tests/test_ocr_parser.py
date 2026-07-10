"""Unit tests for modules.ocr.parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.ocr.parser import parse_ocr_text, text_similarity


def test_parse_dimension_with_unit():
    result = parse_ocr_text("450 mm")
    assert result.category == "dimension"
    assert result.numeric_value == 450.0
    assert result.unit == "mm"


def test_parse_elevation():
    result = parse_ocr_text("EL. +12.500")
    assert result.category == "elevation"
    assert result.numeric_value == 12.5


def test_parse_level():
    result = parse_ocr_text("LEVEL 3")
    assert result.category == "level"


def test_parse_generic_note():
    result = parse_ocr_text("SEE DETAIL A FOR REINFORCEMENT")
    assert result.category == "note"


def test_parse_empty_string():
    result = parse_ocr_text("")
    assert result.category == "unknown"


def test_text_similarity_identical():
    assert text_similarity("BEAM B-12", "BEAM B-12") == 1.0


def test_text_similarity_completely_different():
    sim = text_similarity("BEAM B-12", "COLUMN C-4")
    assert 0.0 <= sim < 0.6


def test_text_similarity_minor_difference():
    sim = text_similarity("450 mm", "455 mm")
    assert sim > 0.7
