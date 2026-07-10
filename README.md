# 📐 AI-Powered CAD Revision Analyzer

A production-grade, modular pipeline that automatically detects meaningful
engineering changes between two versions (**BEFORE** / **AFTER**) of a CAD
drawing PDF, using a **hybrid** Computer Vision + OCR + Object Detection +
Matching + Rule-Based Reasoning pipeline (never SSIM alone).

It outputs numbered, color-coded annotated images, a JSON report, a
Markdown report, and ships with a full Streamlit web UI.

---

## ✨ What it detects

| Category | Meaning |
|---|---|
| 🟢 **Added** | Present in AFTER, not in BEFORE |
| 🔴 **Removed** | Present in BEFORE, not in AFTER |
| 🟡 **Modified** | Multiple signals changed at once (geometry + text/dimension) |
| 🟡 **Dimension Change** | A numeric dimension value changed |
| 🟡 **Text Change** | Note/annotation text changed |
| 🔵 **Geometry Change** | Shape/position changed but text/dimension are stable |

Every detection carries a **confidence score**, a **bounding box**, and a
human-readable **reason**.

---

## 🏗️ Architecture

```
cad_revision_ai/
├── app/
│   ├── main.py          # CLI entrypoint
│   └── pipeline.py       # 10-stage orchestrator
├── config/
│   └── config.py         # ALL tunable parameters (no hardcoded values anywhere else)
├── data/
│   ├── input/  output/  temp/
├── modules/
│   ├── pdf/               # Stage 1 — render.py, vector_extract.py
│   ├── preprocess/        # Stage 2 — deskew, denoise, binarize, roi
│   ├── alignment/         # Stage 3 — ecc.py (sub-pixel ECC), refine.py
│   ├── detection/         # Stage 4 — ssim, edges, morphology, components, clustering
│   ├── ocr/                # Stage 5 — preprocess, tesseract, parser
│   ├── yolo/               # Stage 6 — detector.py, trainer.py
│   ├── matching/           # Stage 7 — hungarian.py, matcher.py
│   ├── classification/     # Stage 8 — classify.py
│   ├── visualization/      # Stage 9 — draw_boxes.py, legend.py
│   ├── reporting/          # Stage 10 — report_json.py, report_md.py
│   └── utils/               # geometry.py, logger.py (shared helpers)
├── models/
│   └── yolov8_custom.pt    # (you train this — see below)
├── streamlit_app.py         # Full-featured web UI
├── requirements.txt
└── README.md
```

Every module has type hints, docstrings, logging, and error handling, and
every tunable numeric/string parameter lives in `config/config.py` — no
hardcoded constants elsewhere.

---

## 🔬 Pipeline stages

1. **Render** — PyMuPDF renders each PDF page at 600 DPI.
2. **Preprocess** — denoise (Non-Local Means) → deskew (projection-profile) →
   border removal → auto-crop to drawing area.
3. **Alignment** — **ECC** (Enhanced Correlation Coefficient), run on an
   image pyramid then refined at full resolution, followed by phase-correlation
   sub-pixel refinement. ORB/SIFT/AKAZE are deliberately **not** used — they
   frequently mismatch on the sparse, repetitive line-work typical of CAD
   drawings.
4. **Difference detection** — SSIM + Canny edges are fused into one mask,
   cleaned with morphological opening/closing, then connected-components +
   contour filtering + DBSCAN clustering produce the final candidate regions.
5. **OCR** — Tesseract, with ROI-specific upscaling/sharpening/thresholding,
   PSM 6 (notes) / PSM 7 (dimensions), and an engineering character
   whitelist. A regex-based parser extracts dimensions, elevations, and levels.
6. **Object detection** — Custom YOLOv8 with CAD-specific classes (beam,
   column, pier, pile, wall, road, text_box, dimension, arrow, leader, note,
   annotation, foundation, utility) — **not COCO classes**.
7. **Matching** — Hungarian (Kuhn-Munkres) optimal assignment over a
   multi-factor weighted cost matrix (distance, type, dimension, direction,
   geometry, text similarity, IoU).
8. **Classification** — Rule engine turns matched/unmatched pairs into
   final change categories with confidence scores.
9. **Visualization** — Numbered, color-coded boxes + legend + confidence
   labels, overlapping boxes merged for a clean result.
10. **Reporting** — JSON (machine-readable) + Markdown (human-readable) +
    annotated PNG.

---

## ⚠️ About the YOLO model

This repo ships the **full training pipeline** (`modules/yolo/trainer.py`)
but **not** pretrained weights for the custom class list — that requires a
labeled CAD-drawing dataset that only you can provide (label with CVAT /
Roboflow / LabelImg in YOLO format).

**Until you train and drop `models/yolov8_custom.pt` in place, the pipeline
still works end-to-end** — it just runs without semantic object classes,
relying on the classical CV + OCR + geometric matching stages, which are
fully functional out of the box. Once trained weights are present,
`YOLODetector` picks them up automatically with zero code changes.

To train:
```bash
python -m modules.yolo.trainer --data path/to/dataset_root
```

---

## 🚀 Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Tesseract OCR (system binary, not pip-installable)
#    Ubuntu/Debian: sudo apt-get install tesseract-ocr
#    macOS:         brew install tesseract
#    Windows:       https://github.com/UB-Mannheim/tesseract/wiki

# 4. (Optional) verify GPU/CPU torch install for YOLO
python -c "import torch; print(torch.__version__)"
```

---

## ▶️ Usage

### CLI
```bash
python -m app.main --before drawings/before.pdf --after drawings/after.pdf \
    --output data/output/run_001 --page 0
```

### Streamlit Web App
```bash
streamlit run streamlit_app.py
```
Then open the printed local URL, upload your BEFORE/AFTER PDFs, and click
**Run Revision Analysis**. You'll get:
- A live dashboard with summary metrics
- The annotated comparison image with legend
- A filterable change table + card view + charts
- One-click JSON / Markdown / PNG downloads

---

## 🔧 Configuration

Every parameter can be overridden via environment variables prefixed with
`CAD_`, e.g.:
```bash
export CAD_RENDER_DPI=800
export CAD_ECC_MOTION_TYPE=AFFINE
export CAD_MATCH_SCORE_THRESHOLD=0.4
```
See `config/config.py` for the full list of ~70 tunable parameters across
PDF rendering, preprocessing, alignment, detection, OCR, YOLO, matching,
classification, visualization, and reporting.

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=modules --cov=app
```

---

## 📌 Design notes

- **No single point of failure**: every external dependency (Tesseract,
  YOLO weights) degrades gracefully with a logged warning rather than
  crashing the pipeline.
- **SOLID principles**: each module has a single responsibility; the
  pipeline orchestrator composes them without duplicating logic.
- **No hardcoded values**: all thresholds/weights live in `config.py`.
- **Sub-pixel accuracy**: ECC + phase-correlation refinement, not a single
  coarse alignment pass.
