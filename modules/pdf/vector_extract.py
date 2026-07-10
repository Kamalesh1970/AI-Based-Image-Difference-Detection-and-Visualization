"""
vector_extract.py
==================
Extracts native vector geometry (lines, curves, rectangles) and text spans
directly from the PDF's content stream, when available. This provides a
ground-truth companion signal to the raster-based CV pipeline: many CAD
exports embed real vector paths and text objects rather than flattened
images, and reading them directly is far more precise than OCR/edge
detection when it's possible.

If a PDF has no extractable vector content (i.e. it's a scanned raster
embedded as an image), this module degrades gracefully and returns empty
results; the raster pipeline (render.py + CV stages) remains the source of
truth in that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import fitz  # PyMuPDF

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VectorTextSpan:
    text: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in PDF points
    font: str
    size: float


@dataclass
class VectorPath:
    kind: str  # 'line' | 'rect' | 'curve' | 'other'
    points: List[Tuple[float, float]]
    bbox: Tuple[float, float, float, float]
    stroke_width: float = 0.0


@dataclass
class VectorPageData:
    page_index: int
    text_spans: List[VectorTextSpan] = field(default_factory=list)
    paths: List[VectorPath] = field(default_factory=list)
    has_vector_content: bool = False


class VectorExtractor:
    """Extracts vector primitives from PDF pages using PyMuPDF's low-level APIs."""

    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: multiply extracted coordinates by this factor to align
                   with a raster image rendered at a given DPI
                   (scale = dpi / 72.0).
        """
        self.scale = scale

    def extract(self, pdf_path: Union[str, Path], max_pages: int | None = None) -> List[VectorPageData]:
        if not settings.pdf.extract_vector_data:
            logger.info("Vector extraction disabled via config; skipping.")
            return []

        pdf_path = Path(pdf_path)
        max_pages = max_pages or settings.pdf.max_pages
        results: List[VectorPageData] = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector extraction failed to open '%s': %s", pdf_path, exc)
            return []

        try:
            n_pages = min(max_pages, doc.page_count)
            for page_index in range(n_pages):
                page = doc.load_page(page_index)
                page_data = VectorPageData(page_index=page_index)

                # --- Text spans ---
                try:
                    raw = page.get_text("dict")
                    for block in raw.get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue
                                bx = span.get("bbox", (0, 0, 0, 0))
                                page_data.text_spans.append(
                                    VectorTextSpan(
                                        text=text,
                                        bbox=self._scaled_bbox(bx),
                                        font=span.get("font", ""),
                                        size=float(span.get("size", 0.0)),
                                    )
                                )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Text span extraction issue on page %d: %s", page_index, exc)

                # --- Vector drawing paths ---
                try:
                    drawings = page.get_drawings()
                    for d in drawings:
                        for item in d.get("items", []):
                            kind = item[0]
                            pts: List[Tuple[float, float]] = []
                            if kind == "l":  # line
                                p1, p2 = item[1], item[2]
                                pts = [(p1.x, p1.y), (p2.x, p2.y)]
                                path_kind = "line"
                            elif kind == "re":  # rectangle
                                rect = item[1]
                                pts = [
                                    (rect.x0, rect.y0), (rect.x1, rect.y0),
                                    (rect.x1, rect.y1), (rect.x0, rect.y1),
                                ]
                                path_kind = "rect"
                            elif kind == "c":  # bezier curve
                                pts = [(p.x, p.y) for p in item[1:] if hasattr(p, "x")]
                                path_kind = "curve"
                            else:
                                continue

                            if not pts:
                                continue
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            bbox = (min(xs), min(ys), max(xs), max(ys))
                            page_data.paths.append(
                                VectorPath(
                                    kind=path_kind,
                                    points=[self._scaled_point(p) for p in pts],
                                    bbox=self._scaled_bbox(bbox),
                                    stroke_width=float(d.get("width") or 0.0),
                                )
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Drawing extraction issue on page %d: %s", page_index, exc)

                page_data.has_vector_content = bool(page_data.text_spans or page_data.paths)
                logger.info(
                    "Page %d vector extraction: %d text spans, %d paths (vector_content=%s)",
                    page_index, len(page_data.text_spans), len(page_data.paths),
                    page_data.has_vector_content,
                )
                results.append(page_data)
        finally:
            doc.close()

        return results

    def _scaled_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale)

    def _scaled_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        return (point[0] * self.scale, point[1] * self.scale)
