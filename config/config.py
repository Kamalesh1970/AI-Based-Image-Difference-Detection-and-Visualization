"""
config.py
=========
Central configuration for the AI-Powered CAD Revision Analyzer.

Every tunable parameter in the pipeline is defined here so that no module
contains hardcoded magic numbers. Values can be overridden at runtime via
environment variables (prefixed with CAD_) without touching source code.

Author: CAD Revision AI Team
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"CAD_{name}", default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"CAD_{name}", default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"CAD_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(f"CAD_{name}")
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(init=False)
    input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    temp_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "data_dir", self.project_root / "data")
        object.__setattr__(self, "input_dir", self.data_dir / "input")
        object.__setattr__(self, "output_dir", self.data_dir / "output")
        object.__setattr__(self, "temp_dir", self.data_dir / "temp")
        object.__setattr__(self, "models_dir", self.project_root / "models")
        object.__setattr__(self, "log_dir", self.project_root / "logs")
        for p in (self.data_dir, self.input_dir, self.output_dir,
                  self.temp_dir, self.models_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# PDF rendering configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PDFConfig:
    render_dpi: int = _env_int("RENDER_DPI", 600)
    color_space: str = _env_str("PDF_COLORSPACE", "RGB")
    alpha: bool = False
    max_pages: int = _env_int("PDF_MAX_PAGES", 1)  # compare first N pages by default
    extract_vector_data: bool = _env_bool("PDF_EXTRACT_VECTOR", True)


# --------------------------------------------------------------------------- #
# Preprocessing configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreprocessConfig:
    denoise_h: float = _env_float("DENOISE_H", 7.0)
    denoise_template_window: int = _env_int("DENOISE_TEMPLATE_WINDOW", 7)
    denoise_search_window: int = _env_int("DENOISE_SEARCH_WINDOW", 21)

    adaptive_thresh_block_size: int = _env_int("ADAPTIVE_BLOCK_SIZE", 35)
    adaptive_thresh_c: int = _env_int("ADAPTIVE_C", 11)

    deskew_angle_limit: float = _env_float("DESKEW_ANGLE_LIMIT", 10.0)
    deskew_angle_step: float = _env_float("DESKEW_ANGLE_STEP", 0.1)

    border_crop_px: int = _env_int("BORDER_CROP_PX", 15)
    auto_crop_drawing_area: bool = _env_bool("AUTO_CROP_DRAWING", True)
    roi_margin_ratio: float = _env_float("ROI_MARGIN_RATIO", 0.01)


# --------------------------------------------------------------------------- #
# Alignment configuration (ECC based - subpixel)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlignmentConfig:
    motion_type: str = _env_str("ECC_MOTION_TYPE", "HOMOGRAPHY")  # TRANSLATION/EUCLIDEAN/AFFINE/HOMOGRAPHY
    number_of_iterations: int = _env_int("ECC_ITERATIONS", 5000)
    termination_eps: float = _env_float("ECC_EPS", 1e-8)
    gaussian_blur_size: int = _env_int("ECC_GAUSS_BLUR", 5)
    pyramid_levels: int = _env_int("ECC_PYRAMID_LEVELS", 3)
    downscale_for_ecc: float = _env_float("ECC_DOWNSCALE", 0.5)  # speeds up + stabilizes ECC
    refine_with_phase_correlation: bool = _env_bool("ECC_PHASE_REFINE", True)


# --------------------------------------------------------------------------- #
# Difference detection configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectionConfig:
    ssim_win_size: int = _env_int("SSIM_WIN_SIZE", 11)
    ssim_diff_threshold: float = _env_float("SSIM_DIFF_THRESHOLD", 0.85)

    canny_low: int = _env_int("CANNY_LOW", 50)
    canny_high: int = _env_int("CANNY_HIGH", 150)

    morph_kernel_size: int = _env_int("MORPH_KERNEL_SIZE", 5)
    morph_open_iterations: int = _env_int("MORPH_OPEN_ITER", 1)
    morph_close_iterations: int = _env_int("MORPH_CLOSE_ITER", 2)

    min_component_area: int = _env_int("MIN_COMPONENT_AREA", 40)
    max_component_area_ratio: float = _env_float("MAX_COMPONENT_AREA_RATIO", 0.5)

    cluster_merge_distance_px: int = _env_int("CLUSTER_MERGE_DIST", 25)
    dbscan_eps: float = _env_float("DBSCAN_EPS", 30.0)
    dbscan_min_samples: int = _env_int("DBSCAN_MIN_SAMPLES", 1)

    contour_min_perimeter: float = _env_float("CONTOUR_MIN_PERIMETER", 10.0)
    box_padding_px: int = _env_int("BOX_PADDING_PX", 6)


# --------------------------------------------------------------------------- #
# OCR configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OCRConfig:
    tesseract_cmd: str = _env_str("TESSERACT_CMD", "tesseract")
    lang: str = _env_str("OCR_LANG", "eng")
    psm_dimension: int = _env_int("OCR_PSM_DIMENSION", 7)  # single line
    psm_note: int = _env_int("OCR_PSM_NOTE", 6)  # block of text
    oem: int = _env_int("OCR_OEM", 3)
    upscale_factor: float = _env_float("OCR_UPSCALE_FACTOR", 3.0)
    sharpen_strength: float = _env_float("OCR_SHARPEN_STRENGTH", 1.5)
    char_whitelist: str = _env_str(
        "OCR_WHITELIST",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        ".,:;'\"-+#@°ØøΦ/\\()[]±=%",
    )
    min_confidence: float = _env_float("OCR_MIN_CONFIDENCE", 35.0)
    dimension_regex: str = r"(\d+[.,]?\d*)\s*(mm|cm|m|ft|in|\"|')?"
    elevation_regex: str = r"\b(ELEV|EL|RL)\b[\s.:]*([+-]?\d+[.,]?\d*)"
    level_regex: str = r"\b(LEVEL|LVL|FL)\b[\s.:]*(\d+|[A-Z0-9\-]+)"


# --------------------------------------------------------------------------- #
# YOLO configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class YOLOConfig:
    weights_path: str = _env_str("YOLO_WEIGHTS", "models/yolov8_custom.pt")
    fallback_weights: str = _env_str("YOLO_FALLBACK_WEIGHTS", "yolov8n.pt")
    confidence_threshold: float = _env_float("YOLO_CONF_THRESH", 0.25)
    iou_threshold: float = _env_float("YOLO_IOU_THRESH", 0.45)
    image_size: int = _env_int("YOLO_IMGSZ", 1280)
    device: str = _env_str("YOLO_DEVICE", "cpu")
    class_names: Tuple[str, ...] = (
        "beam", "column", "pier", "pile", "wall", "road",
        "text_box", "dimension", "arrow", "leader", "note",
        "annotation", "foundation", "utility",
    )
    enabled: bool = _env_bool("YOLO_ENABLED", True)
    training_epochs: int = _env_int("YOLO_TRAIN_EPOCHS", 100)
    training_batch: int = _env_int("YOLO_TRAIN_BATCH", 16)


# --------------------------------------------------------------------------- #
# Matching configuration (Hungarian algorithm weights)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatchingConfig:
    weight_distance: float = _env_float("MATCH_W_DISTANCE", 0.25)
    weight_type: float = _env_float("MATCH_W_TYPE", 0.20)
    weight_dimension: float = _env_float("MATCH_W_DIMENSION", 0.15)
    weight_direction: float = _env_float("MATCH_W_DIRECTION", 0.10)
    weight_geometry: float = _env_float("MATCH_W_GEOMETRY", 0.10)
    weight_text: float = _env_float("MATCH_W_TEXT", 0.10)
    weight_iou: float = _env_float("MATCH_W_IOU", 0.10)

    max_match_distance_px: float = _env_float("MAX_MATCH_DISTANCE", 250.0)
    unmatched_cost: float = _env_float("UNMATCHED_COST", 1.0)
    match_score_threshold: float = _env_float("MATCH_SCORE_THRESHOLD", 0.35)


# --------------------------------------------------------------------------- #
# Classification configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassificationConfig:
    dimension_change_tolerance: float = _env_float("DIM_CHANGE_TOLERANCE", 0.01)
    text_similarity_threshold: float = _env_float("TEXT_SIM_THRESHOLD", 0.90)
    geometry_iou_threshold: float = _env_float("GEOMETRY_IOU_THRESHOLD", 0.92)
    confidence_high: float = _env_float("CONF_HIGH", 0.80)
    confidence_medium: float = _env_float("CONF_MEDIUM", 0.55)


# --------------------------------------------------------------------------- #
# Visualization configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VisualizationConfig:
    color_added: Tuple[int, int, int] = (0, 200, 0)        # Green (BGR)
    color_removed: Tuple[int, int, int] = (0, 0, 220)       # Red
    color_modified: Tuple[int, int, int] = (0, 200, 220)    # Yellow
    color_geometry: Tuple[int, int, int] = (220, 130, 0)    # Blue
    box_thickness: int = _env_int("BOX_THICKNESS", 3)
    font_scale: float = _env_float("FONT_SCALE", 0.55)
    font_thickness: int = _env_int("FONT_THICKNESS", 1)
    label_bg_alpha: float = _env_float("LABEL_BG_ALPHA", 0.65)
    legend_width_px: int = _env_int("LEGEND_WIDTH", 260)
    overlap_merge_iou: float = _env_float("OVERLAP_MERGE_IOU", 0.6)
    id_circle_radius: int = _env_int("ID_CIRCLE_RADIUS", 12)


# --------------------------------------------------------------------------- #
# Reporting configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReportingConfig:
    json_indent: int = _env_int("JSON_INDENT", 2)
    include_thumbnails: bool = _env_bool("REPORT_THUMBNAILS", True)
    report_title: str = _env_str("REPORT_TITLE", "CAD Revision Analysis Report")
    company_name: str = _env_str("COMPANY_NAME", "AI-Powered CAD Revision Analyzer")


# --------------------------------------------------------------------------- #
# Master configuration object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    log_level: str = _env_str("LOG_LEVEL", "INFO")


# Singleton-style accessor. Import `settings` anywhere in the codebase.
settings = AppConfig()
