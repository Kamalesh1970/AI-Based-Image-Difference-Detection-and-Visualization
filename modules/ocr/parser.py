"""
parser.py (ocr)
================
Stage 5c: Structured parsing of raw OCR text into engineering-meaningful
fields: dimension values + units, elevations, levels, and free-form notes.
Uses regex rules defined centrally in config.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedText:
    raw_text: str
    category: str  # 'dimension' | 'elevation' | 'level' | 'note' | 'unknown'
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    normalized_text: str = ""
    matches: List[str] = field(default_factory=list)


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def parse_ocr_text(raw_text: str) -> ParsedText:
    """
    Classify and structurally parse a raw OCR string into a ParsedText
    object. Tries dimension -> elevation -> level patterns in that order,
    falling back to a generic 'note' category.
    """
    cfg = settings.ocr
    text = raw_text.strip()
    normalized = re.sub(r"\s+", " ", text).upper()

    if not text:
        return ParsedText(raw_text=raw_text, category="unknown", normalized_text="")

    # --- Elevation (EL / ELEV / RL) ---
    elev_match = re.search(cfg.elevation_regex, normalized, re.IGNORECASE)
    if elev_match:
        value = _to_float(elev_match.group(2))
        return ParsedText(
            raw_text=raw_text, category="elevation", numeric_value=value,
            normalized_text=normalized, matches=[elev_match.group(0)],
        )

    # --- Level (LEVEL / LVL / FL) ---
    level_match = re.search(cfg.level_regex, normalized, re.IGNORECASE)
    if level_match:
        return ParsedText(
            raw_text=raw_text, category="level",
            normalized_text=normalized, matches=[level_match.group(0)],
        )

    # --- Dimension (numeric value + optional unit) ---
    dim_matches = re.findall(cfg.dimension_regex, normalized, re.IGNORECASE)
    if dim_matches:
        # Use the first numeric match as the primary value
        value_str, unit = dim_matches[0]
        value = _to_float(value_str)
        if value is not None:
            return ParsedText(
                raw_text=raw_text, category="dimension", numeric_value=value,
                unit=(unit.lower() or None) if unit else None, normalized_text=normalized,
                matches=[f"{m[0]}{m[1]}" for m in dim_matches],
            )

    # --- Fallback: general note/annotation text ---
    return ParsedText(raw_text=raw_text, category="note", normalized_text=normalized)


def text_similarity(a: str, b: str) -> float:
    """
    Normalized Levenshtein-based similarity in [0, 1], 1 = identical.
    Uses python-Levenshtein when available, otherwise a pure-python fallback.
    """
    a_norm = re.sub(r"\s+", " ", a.strip().upper())
    b_norm = re.sub(r"\s+", " ", b.strip().upper())

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0

    try:
        import Levenshtein
        distance = Levenshtein.distance(a_norm, b_norm)
    except ImportError:
        distance = _levenshtein_fallback(a_norm, b_norm)

    max_len = max(len(a_norm), len(b_norm))
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


def _levenshtein_fallback(a: str, b: str) -> int:
    """Pure-python Levenshtein distance, used only if the C extension isn't installed."""
    if len(a) < len(b):
        return _levenshtein_fallback(b, a)
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
