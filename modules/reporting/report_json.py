"""
report_json.py
==============
Stage 10a: Generates a structured JSON report describing every detected
change, plus run metadata and summary statistics, suitable for
machine-to-machine consumption (e.g. feeding into a PM/BIM tool).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from config.config import settings
from modules.classification.classify import ClassifiedChange
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def _change_to_dict(idx: int, change: ClassifiedChange) -> Dict[str, Any]:
    return {
        "id": idx,
        "region_id": change.id,
        "category": change.category.value,
        "confidence": round(change.confidence, 4),
        "object_class": change.object_class,
        "source": change.source,
        "bounding_box": {
            "x1": change.bbox.x1, "y1": change.bbox.y1,
            "x2": change.bbox.x2, "y2": change.bbox.y2,
            "width": change.bbox.width, "height": change.bbox.height,
        },
        "before_text": change.before_text,
        "after_text": change.after_text,
        "before_value": change.before_value,
        "after_value": change.after_value,
        "reasons": change.reasons,
    }


def build_report_dict(
    changes: List[ClassifiedChange],
    before_pdf: str,
    after_pdf: str,
    ssim_score: float,
    alignment_correlation: float,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble the full report payload as a plain Python dict."""
    counts = Counter(c.category.value for c in changes)
    return {
        "report_title": settings.reporting.report_title,
        "generated_by": settings.reporting.company_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "before_pdf": before_pdf,
            "after_pdf": after_pdf,
        },
        "quality_metrics": {
            "global_ssim_score": round(ssim_score, 4),
            "alignment_correlation": round(alignment_correlation, 4),
        },
        "summary": {
            "total_changes": len(changes),
            "by_category": dict(counts),
        },
        "changes": [_change_to_dict(i, c) for i, c in enumerate(changes, start=1)],
        "metadata": extra_metadata or {},
    }


def save_json_report(report: Dict[str, Any], output_path: Union[str, Path]) -> str:
    """Write the report dict to disk as pretty-printed JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=settings.reporting.json_indent, ensure_ascii=False)
    logger.info("JSON report saved to '%s' (%d changes).", output_path, len(report.get("changes", [])))
    return str(output_path)
