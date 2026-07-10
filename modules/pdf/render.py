"""
render.py
=========
Stage 1: Render engineering-drawing PDFs to high-resolution raster images
using PyMuPDF (fitz). Rendering at 600 DPI preserves fine line-work and
small text so that downstream CV/OCR stages have maximum detail to work
with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF
import numpy as np

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


class PDFRenderError(RuntimeError):
    """Raised when a PDF cannot be opened or rendered."""


@dataclass
class RenderedPage:
    page_index: int
    image_bgr: np.ndarray
    dpi: int
    width_px: int
    height_px: int
    source_path: str


class PDFRenderer:
    """Renders PDF pages to OpenCV-compatible (BGR, uint8) numpy arrays."""

    def __init__(self, dpi: int | None = None):
        self.dpi = dpi or settings.pdf.render_dpi

    def render(self, pdf_path: Union[str, Path], max_pages: int | None = None) -> List[RenderedPage]:
        """
        Render the first `max_pages` pages of a PDF at the configured DPI.

        Args:
            pdf_path: path to the source PDF.
            max_pages: number of pages to render (defaults to config value).

        Returns:
            List of RenderedPage objects, one per rendered page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFRenderError(f"PDF not found: {pdf_path}")

        max_pages = max_pages or settings.pdf.max_pages
        zoom = self.dpi / 72.0  # PDF base resolution is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)

        rendered: List[RenderedPage] = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            raise PDFRenderError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

        try:
            n_pages = min(max_pages, doc.page_count)
            if n_pages == 0:
                raise PDFRenderError(f"PDF '{pdf_path}' has no pages.")

            for page_index in range(n_pages):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix, alpha=settings.pdf.alpha)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                # Convert RGB(A) -> BGR for OpenCV consistency
                if pix.n == 4:
                    img_bgr = img[:, :, [2, 1, 0]]
                elif pix.n == 3:
                    img_bgr = img[:, :, ::-1]
                else:
                    img_bgr = np.stack([img[:, :, 0]] * 3, axis=-1)

                rendered.append(
                    RenderedPage(
                        page_index=page_index,
                        image_bgr=np.ascontiguousarray(img_bgr),
                        dpi=self.dpi,
                        width_px=pix.width,
                        height_px=pix.height,
                        source_path=str(pdf_path),
                    )
                )
                logger.info(
                    "Rendered page %d of '%s' at %d DPI -> %dx%d px",
                    page_index, pdf_path.name, self.dpi, pix.width, pix.height,
                )
        finally:
            doc.close()

        return rendered

    def render_single(self, pdf_path: Union[str, Path], page_index: int = 0) -> RenderedPage:
        """Convenience method returning only one page."""
        pages = self.render(pdf_path, max_pages=page_index + 1)
        if page_index >= len(pages):
            raise PDFRenderError(f"Page index {page_index} out of range for '{pdf_path}'.")
        return pages[page_index]
