"""
pipeline.py
===========
The orchestrator that wires together all 10 stages of the AI-Powered CAD
Revision Analyzer into a single, callable, end-to-end pipeline:

  1. PDF rendering              (modules.pdf.render)
  2. Preprocessing              (modules.preprocess.*)
  3. Alignment                  (modules.alignment.*)
  4. Difference detection       (modules.detection.*)
  5. OCR                        (modules.ocr.*)
  6. Object detection (YOLO)    (modules.yolo.detector)
  7. Matching                   (modules.matching.matcher)
  8. Classification             (modules.classification.classify)
  9. Visualization              (modules.visualization.*)
 10. Reporting                  (modules.reporting.*)

Usage:
    from app.pipeline import CADRevisionPipeline
    pipeline = CADRevisionPipeline()
    result = pipeline.run("before.pdf", "after.pdf", output_dir="data/output/run1")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from config.config import settings
from modules.alignment.ecc import align_ecc
from modules.alignment.refine import compute_alignment_quality, refine_subpixel
from modules.classification.classify import ClassifiedChange, classify_changes
from modules.detection.clustering import cluster_candidates
from modules.detection.components import CandidateRegion, detect_candidate_regions
from modules.matching.matcher import match_regions
from modules.ocr.parser import parse_ocr_text
from modules.ocr.tesseract import OCRError, is_tesseract_available, run_ocr
from modules.pdf.render import PDFRenderer
from modules.preprocess.binarize import remove_border
from modules.preprocess.deskew import deskew_image
from modules.preprocess.denoise import denoise_image
from modules.preprocess.roi import auto_crop_drawing_area
from modules.reporting.report_json import build_report_dict, save_json_report
from modules.reporting.report_md import save_markdown_report
from modules.utils.geometry import BoundingBox, DetectedRegion
from modules.utils.logger import get_logger
from modules.visualization.draw_boxes import draw_annotated_image, merge_same_category_overlaps
from modules.visualization.legend import append_legend
from modules.yolo.detector import YOLODetector

logger = get_logger(__name__)


@dataclass
class StageTiming:
    stage: str
    seconds: float


@dataclass
class PipelineResult:
    changes: List[ClassifiedChange]
    annotated_image: np.ndarray
    annotated_image_with_legend: np.ndarray
    report: dict
    json_report_path: str
    markdown_report_path: str
    annotated_image_path: str
    ssim_score: float
    alignment_correlation: float
    alignment_quality: float
    timings: List[StageTiming] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CADRevisionPipeline:
    """End-to-end orchestrator for the CAD revision-analysis pipeline."""

    def __init__(self):
        self.renderer = PDFRenderer()
        self.yolo_detector = YOLODetector()
        self._tesseract_ok = is_tesseract_available()
        if not self._tesseract_ok:
            logger.warning(
                "Tesseract is not available on this system. OCR-derived signals "
                "(dimension/text change detection) will be skipped; geometry-based "
                "detection continues to function normally."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        before_pdf: Union[str, Path],
        after_pdf: Union[str, Path],
        output_dir: Union[str, Path],
        page_index: int = 0,
    ) -> PipelineResult:
        """
        Execute the full pipeline on one page of a before/after PDF pair.

        Args:
            before_pdf: path to the BEFORE drawing PDF.
            after_pdf: path to the AFTER drawing PDF.
            output_dir: directory where reports/images will be written.
            page_index: which page to compare (0-indexed).

        Returns:
            PipelineResult with all outputs and file paths.
        """
        timings: List[StageTiming] = []
        warnings: List[str] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: Render ---
        t0 = time.time()
        before_page = self.renderer.render_single(before_pdf, page_index)
        after_page = self.renderer.render_single(after_pdf, page_index)
        timings.append(StageTiming("1_render_pdf", time.time() - t0))

        # --- Stage 2: Preprocess ---
        t0 = time.time()
        before_pre, before_roi = self._preprocess(before_page.image_bgr)
        after_pre, after_roi = self._preprocess(after_page.image_bgr)
        timings.append(StageTiming("2_preprocess", time.time() - t0))

        # --- Stage 3: Alignment ---
        t0 = time.time()
        alignment = align_ecc(before_pre, after_pre)
        refinement = refine_subpixel(before_pre, alignment.aligned_image)
        aligned_after = refinement.refined_image
        alignment_quality = compute_alignment_quality(before_pre, aligned_after)
        timings.append(StageTiming("3_alignment", time.time() - t0))

        if not alignment.converged or alignment_quality < 0.4:
            warnings.append(
                f"Alignment quality is low (score={alignment_quality:.2f}). "
                "Results may include false positives caused by residual misalignment."
            )

        # --- Stage 4: Difference detection (SSIM + Canny + morphology + CC + contours) ---
        t0 = time.time()
        fusion = detect_candidate_regions(before_pre, aligned_after)
        merged_candidates = cluster_candidates(fusion.candidates)
        timings.append(StageTiming("4_difference_detection", time.time() - t0))

        # --- Stage 6: YOLO object detection (runs on both pages independently) ---
        t0 = time.time()
        before_yolo = self.yolo_detector.detect(before_pre)
        after_yolo = self.yolo_detector.detect(aligned_after)
        if not self.yolo_detector.is_active:
            warnings.append(
                "YOLO custom model is not active (no trained weights found). "
                "Object-class-aware detection is disabled; classical CV-based "
                "geometric change detection is still fully active."
            )
        timings.append(StageTiming("6_yolo_detection", time.time() - t0))

        # --- Build unified DetectedRegion lists for BEFORE/AFTER ---
        before_regions = self._build_regions(merged_candidates, before_yolo.regions, before_pre, side="before")
        after_regions = self._build_regions(merged_candidates, after_yolo.regions, aligned_after, side="after")

        # --- Stage 5: OCR on every candidate region ---
        t0 = time.time()
        if self._tesseract_ok:
            self._run_ocr_on_regions(before_regions, before_pre)
            self._run_ocr_on_regions(after_regions, aligned_after)
        timings.append(StageTiming("5_ocr", time.time() - t0))

        # --- Stage 7: Matching (Hungarian) ---
        t0 = time.time()
        matching_result = match_regions(before_regions, after_regions)
        timings.append(StageTiming("7_matching", time.time() - t0))

        # --- Stage 8: Classification ---
        t0 = time.time()
        changes = classify_changes(matching_result)
        changes = merge_same_category_overlaps(changes)
        changes.sort(key=lambda c: (c.bbox.y1, c.bbox.x1))
        timings.append(StageTiming("8_classification", time.time() - t0))

        # --- Stage 9: Visualization ---
        t0 = time.time()
        annotated = draw_annotated_image(aligned_after, changes)
        annotated_with_legend = append_legend(annotated, changes)
        annotated_path = output_dir / "annotated_comparison.png"
        cv2.imwrite(str(annotated_path), annotated_with_legend)
        timings.append(StageTiming("9_visualization", time.time() - t0))

        # --- Stage 10: Reporting ---
        t0 = time.time()
        report = build_report_dict(
            changes=changes,
            before_pdf=str(before_pdf),
            after_pdf=str(after_pdf),
            ssim_score=fusion.ssim_score,
            alignment_correlation=alignment.correlation_coefficient,
            extra_metadata={
                "yolo_active": self.yolo_detector.is_active,
                "tesseract_available": self._tesseract_ok,
                "alignment_quality": round(alignment_quality, 4),
                "warnings": warnings,
                "stage_timings_seconds": {t.stage: round(t.seconds, 3) for t in timings},
            },
        )
        json_path = save_json_report(report, output_dir / "report.json")
        md_path = save_markdown_report(report, output_dir / "report.md", annotated_image_relpath="annotated_comparison.png")
        timings.append(StageTiming("10_reporting", time.time() - t0))

        logger.info(
            "Pipeline run complete: %d changes detected, total time=%.2fs",
            len(changes), sum(t.seconds for t in timings),
        )

        return PipelineResult(
            changes=changes,
            annotated_image=annotated,
            annotated_image_with_legend=annotated_with_legend,
            report=report,
            json_report_path=json_path,
            markdown_report_path=md_path,
            annotated_image_path=str(annotated_path),
            ssim_score=fusion.ssim_score,
            alignment_correlation=alignment.correlation_coefficient,
            alignment_quality=alignment_quality,
            timings=timings,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _preprocess(self, image: np.ndarray):
        """Stage 2 pipeline: denoise -> deskew -> border removal -> auto-crop ROI."""
        denoised = denoise_image(image)
        deskewed, _ = deskew_image(denoised)
        bordered = remove_border(deskewed)
        cropped, roi = auto_crop_drawing_area(bordered)
        return cropped, roi

    def _build_regions(
        self,
        candidates: List[CandidateRegion],
        yolo_regions: List[DetectedRegion],
        image: np.ndarray,
        side: str,
    ) -> List[DetectedRegion]:
        """
        Merge classical-CV candidate regions with YOLO semantic detections
        into one unified DetectedRegion list per side. When a YOLO box
        overlaps a CV candidate, the semantic class is attached to it;
        YOLO boxes with no CV overlap are still included (they may
        represent unchanged-but-relevant objects used for matching context).
        """
        regions: List[DetectedRegion] = []
        used_yolo_idx = set()

        for cand in candidates:
            region = DetectedRegion(bbox=cand.bbox, source="cv", detector_confidence=cand.fused_score)
            best_iou = 0.0
            best_idx = -1
            for idx, yr in enumerate(yolo_regions):
                iou = cand.bbox.iou(yr.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou > 0.3:
                region.object_class = yolo_regions[best_idx].object_class
                region.detector_confidence = max(region.detector_confidence, yolo_regions[best_idx].detector_confidence)
                used_yolo_idx.add(best_idx)
            regions.append(region)

        # Include any high-confidence YOLO detections not already covered,
        # in case classical diffing missed an object with subtle pixel changes
        # but YOLO's semantic model still flags it (only when YOLO is active).
        for idx, yr in enumerate(yolo_regions):
            if idx in used_yolo_idx:
                continue
            regions.append(yr)

        logger.debug("Built %d unified regions for '%s' side.", len(regions), side)
        return regions

    def _run_ocr_on_regions(self, regions: List[DetectedRegion], image: np.ndarray) -> None:
        """Run OCR + structured parsing on every region's cropped ROI in-place."""
        h, w = image.shape[:2]
        for region in regions:
            box: BoundingBox = region.bbox
            x1, y1 = max(0, box.x1), max(0, box.y1)
            x2, y2 = min(w, box.x2), min(h, box.y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            mode = "dimension" if (box.width < 200 and box.height < 60) else "note"
            try:
                ocr_result = run_ocr(roi, mode=mode)
            except OCRError as exc:
                logger.warning("OCR unavailable, skipping text extraction: %s", exc)
                self._tesseract_ok = False
                return

            if not ocr_result.text:
                continue

            parsed = parse_ocr_text(ocr_result.text)
            region.ocr_text = parsed.raw_text
            region.ocr_confidence = ocr_result.confidence
            region.numeric_value = parsed.numeric_value
            region.metadata["ocr_category"] = parsed.category
