"""
detector.py
===========
Stage 6: Custom YOLOv8 object detection for CAD-specific classes (beam,
column, pier, pile, wall, road, text_box, dimension, arrow, leader, note,
annotation, foundation, utility). COCO classes are intentionally NOT used.

IMPORTANT — a note on the shipped weights:
This repository does NOT ship pretrained weights for the custom class list,
because that requires a labeled CAD-drawing dataset that only you (the
domain owner) can provide. `modules/yolo/trainer.py` implements the full
training pipeline for `models/yolov8_custom.pt`. Until that file exists,
`YOLODetector` degrades gracefully: it logs a clear warning and returns an
empty detection list, and the rest of the pipeline (classical CV diffing,
OCR, matching, classification) continues to operate at full strength using
class-agnostic geometric regions instead of semantic labels. Once you train
and drop a `models/yolov8_custom.pt` file in place, detection activates
automatically with zero code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.config import settings
from modules.utils.geometry import BoundingBox, DetectedRegion
from modules.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: dict = {}


@dataclass
class YOLODetectionSummary:
    regions: List[DetectedRegion]
    model_used: str
    weights_path: Optional[str]


class YOLODetector:
    """
    Thin, defensive wrapper around Ultralytics YOLOv8. Loads custom weights
    if present; otherwise disables itself without crashing the pipeline.
    """

    def __init__(self, weights_path: Optional[str] = None):
        cfg = settings.yolo
        self.enabled = cfg.enabled
        self.weights_path = weights_path or cfg.weights_path
        self.class_names = list(cfg.class_names)
        self._model = None
        self._active_weights: Optional[str] = None

        if not self.enabled:
            logger.info("YOLO detection disabled via config (YOLO_ENABLED=false).")
            return

        self._model = self._load_model()

    def _load_model(self):
        cfg = settings.yolo
        weights = Path(self.weights_path)

        cache_key = str(weights)
        if cache_key in _MODEL_CACHE:
            self._active_weights = cache_key
            return _MODEL_CACHE[cache_key]

        try:
            from ultralytics import YOLO  # imported lazily; heavy dependency
        except ImportError:
            logger.warning(
                "ultralytics package not installed; YOLO detection stage will be skipped. "
                "Install with `pip install ultralytics` to enable it."
            )
            return None

        if weights.exists():
            try:
                model = YOLO(str(weights))
                self._active_weights = str(weights)
                _MODEL_CACHE[cache_key] = model
                logger.info("Loaded custom YOLOv8 weights from '%s'.", weights)
                return model
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load custom weights '%s': %s", weights, exc)

        logger.warning(
            "Custom weights not found at '%s'. YOLO semantic detection is INACTIVE "
            "until you train and place `models/yolov8_custom.pt` (see modules/yolo/trainer.py). "
            "Classical CV difference detection continues to run unaffected.",
            weights,
        )
        return None

    @property
    def is_active(self) -> bool:
        return self._model is not None

    def detect(self, image: np.ndarray) -> YOLODetectionSummary:
        """
        Run object detection on a single image.

        Returns:
            YOLODetectionSummary — empty regions list if the model is not
            active, so callers never need special-case branching.
        """
        cfg = settings.yolo
        if not self.is_active:
            return YOLODetectionSummary(regions=[], model_used="none", weights_path=None)

        try:
            results = self._model.predict(
                source=image,
                conf=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                imgsz=cfg.image_size,
                device=cfg.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("YOLO inference failed: %s. Returning no detections.", exc)
            return YOLODetectionSummary(regions=[], model_used="error", weights_path=self._active_weights)

        regions: List[DetectedRegion] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                cls_idx = int(box.cls[0]) if box.cls is not None else -1
                cls_name = (
                    self.class_names[cls_idx]
                    if 0 <= cls_idx < len(self.class_names)
                    else f"class_{cls_idx}"
                )
                regions.append(
                    DetectedRegion(
                        bbox=BoundingBox(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        source="yolo",
                        object_class=cls_name,
                        detector_confidence=conf,
                    )
                )

        logger.info("YOLO detected %d objects using weights '%s'.", len(regions), self._active_weights)
        return YOLODetectionSummary(
            regions=regions, model_used="yolov8_custom", weights_path=self._active_weights
        )
