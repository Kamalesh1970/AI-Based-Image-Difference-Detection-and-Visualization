"""
main.py
=======
Command-line entrypoint for the AI-Powered CAD Revision Analyzer.

Usage:
    python -m app.main --before drawings/before.pdf --after drawings/after.pdf \
        --output data/output/run_001 --page 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import CADRevisionPipeline  # noqa: E402
from modules.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Powered CAD Revision Analyzer — detect engineering changes between two drawing PDFs."
    )
    parser.add_argument("--before", required=True, help="Path to the BEFORE drawing PDF.")
    parser.add_argument("--after", required=True, help="Path to the AFTER drawing PDF.")
    parser.add_argument("--output", default="data/output/run", help="Output directory for reports/images.")
    parser.add_argument("--page", type=int, default=0, help="Page index to compare (0-indexed).")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)

    if not before_path.exists():
        logger.error("BEFORE file not found: %s", before_path)
        return 1
    if not after_path.exists():
        logger.error("AFTER file not found: %s", after_path)
        return 1

    pipeline = CADRevisionPipeline()
    result = pipeline.run(before_path, after_path, output_dir=args.output, page_index=args.page)

    print("\n================ CAD REVISION ANALYSIS COMPLETE ================")
    print(f"Total changes detected : {len(result.changes)}")
    print(f"Global SSIM score      : {result.ssim_score:.4f}")
    print(f"Alignment quality      : {result.alignment_quality:.4f}")
    print(f"Annotated image        : {result.annotated_image_path}")
    print(f"JSON report             : {result.json_report_path}")
    print(f"Markdown report          : {result.markdown_report_path}")
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print("==================================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
