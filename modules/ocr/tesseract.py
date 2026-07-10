"""
tesseract.py
============
Stage 5b: Tesseract OCR wrapper. Runs text extraction with engineering
character whitelisting and configurable page-segmentation modes (PSM 6 for
multi-line notes, PSM 7 for single-line dimensions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytesseract

from config.config import settings
from modules.ocr.preprocess import add_padding, prepare_roi_for_ocr
from modules.utils.logger import get_logger

logger = get_logger(__name__)

pytesseract.pytesseract.tesseract_cmd = settings.ocr.tesseract_cmd


@dataclass
class OCRResult:
    text: str
    confidence: float
    raw_words: list


class OCRError(RuntimeError):
    """Raised when the Tesseract binary is unavailable or OCR fails."""


def _build_config(psm: int) -> str:
    cfg = settings.ocr
    whitelist = cfg.char_whitelist.replace('"', '\\"')
    return f'--oem {cfg.oem} --psm {psm} -c tessedit_char_whitelist="{whitelist}"'


def run_ocr(
    roi: np.ndarray,
    mode: str = "note",
    preprocess: bool = True,
) -> OCRResult:
    """
    Run Tesseract OCR on a single ROI.

    Args:
        roi: image crop (BGR or grayscale).
        mode: 'dimension' (PSM 7, single line) or 'note' (PSM 6, block).
        preprocess: whether to run the ocr.preprocess pipeline first.

    Returns:
        OCRResult with recognized text, mean word confidence, and raw word data.
    """
    cfg = settings.ocr
    psm = cfg.psm_dimension if mode == "dimension" else cfg.psm_note

    image = prepare_roi_for_ocr(roi) if preprocess else roi
    image = add_padding(image, padding=12)

    tess_config = _build_config(psm)

    try:
        data = pytesseract.image_to_data(
            image, lang=cfg.lang, config=tess_config, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError(
            "Tesseract binary not found. Install it (e.g. `apt-get install tesseract-ocr`) "
            "and/or set CAD_TESSERACT_CMD to its path."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("OCR failed on ROI: %s", exc)
        return OCRResult(text="", confidence=0.0, raw_words=[])

    words = []
    confidences = []
    for i, word in enumerate(data.get("text", [])):
        word = word.strip()
        conf_str = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf_str)
        except (ValueError, TypeError):
            conf = -1.0
        if word and conf >= 0:
            words.append(word)
            confidences.append(conf)

    full_text = " ".join(words).strip()
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    if mean_conf < cfg.min_confidence and full_text:
        logger.debug("Low-confidence OCR result (%.1f): '%s'", mean_conf, full_text)

    return OCRResult(text=full_text, confidence=mean_conf, raw_words=words)


def is_tesseract_available() -> bool:
    """Health-check helper used by main.py / streamlit_app.py at startup."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False
