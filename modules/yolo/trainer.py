"""
trainer.py
==========
Training pipeline for the custom YOLOv8 model used in Stage 6. This trains
a YOLOv8 model from scratch (or fine-tuned from Ultralytics' generic
`yolov8n.pt`/`yolov8s.pt` starting weights) on YOUR labeled CAD dataset,
using the custom class list defined in config.py — NOT COCO classes.

USAGE
-----
1. Label your engineering drawings in YOLO format (e.g. using CVAT,
   Roboflow, or LabelImg) with the classes:
   beam, column, pier, pile, wall, road, text_box, dimension, arrow,
   leader, note, annotation, foundation, utility

2. Organize the dataset as:
       dataset/
         images/train/*.jpg
         images/val/*.jpg
         labels/train/*.txt
         labels/val/*.txt

3. Run:
       python -m modules.yolo.trainer --data dataset/data.yaml

This module also provides `generate_data_yaml()` to auto-create the
Ultralytics `data.yaml` descriptor from config.py's class list, so the
class definitions never drift out of sync between training and inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import yaml

from config.config import settings
from modules.utils.logger import get_logger

logger = get_logger(__name__)


class YOLOTrainingError(RuntimeError):
    """Raised when training cannot proceed (missing deps or dataset)."""


def generate_data_yaml(dataset_root: str, output_path: Optional[str] = None) -> str:
    """
    Auto-generate an Ultralytics-compatible `data.yaml` from the class list
    defined centrally in config.py, guaranteeing training/inference class
    consistency.
    """
    dataset_root_path = Path(dataset_root)
    data = {
        "path": str(dataset_root_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(settings.yolo.class_names)},
    }

    output_path = output_path or str(dataset_root_path / "data.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    logger.info("Generated data.yaml at '%s' with %d classes.", output_path, len(settings.yolo.class_names))
    return output_path


def train_yolo_model(
    data_yaml: str,
    base_weights: str = "yolov8n.pt",
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    project: str = "runs/train",
    name: str = "cad_revision_yolov8",
) -> str:
    """
    Fine-tune a YOLOv8 model on the CAD-specific dataset described by
    `data_yaml`. Saves the best checkpoint and copies it into
    `models/yolov8_custom.pt` so `YOLODetector` picks it up automatically.

    Returns:
        Path to the final `models/yolov8_custom.pt` weights file.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise YOLOTrainingError(
            "ultralytics is not installed. Run `pip install ultralytics` first."
        ) from exc

    if not Path(data_yaml).exists():
        raise YOLOTrainingError(f"data.yaml not found at '{data_yaml}'.")

    cfg = settings.yolo
    epochs = epochs or cfg.training_epochs
    batch = batch or cfg.training_batch
    imgsz = imgsz or cfg.image_size
    device = device or cfg.device

    logger.info(
        "Starting YOLOv8 training: base=%s epochs=%d batch=%d imgsz=%d device=%s",
        base_weights, epochs, batch, imgsz, device,
    )

    model = YOLO(base_weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    if not best_weights.exists():
        raise YOLOTrainingError(f"Training completed but best.pt not found at '{best_weights}'.")

    destination = Path(settings.yolo.weights_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(best_weights.read_bytes())

    logger.info("Training complete. Custom weights copied to '%s'.", destination)
    return str(destination)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the custom CAD-revision YOLOv8 model.")
    parser.add_argument("--data", required=True, help="Path to data.yaml (or dataset root to auto-generate one).")
    parser.add_argument("--base-weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    data_yaml_path = args.data
    if not data_yaml_path.endswith(".yaml"):
        data_yaml_path = generate_data_yaml(data_yaml_path)

    train_yolo_model(
        data_yaml=data_yaml_path,
        base_weights=args.base_weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
