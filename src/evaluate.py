"""
Evaluation gate for YOLO waste classification model.
Only promotes the model if mAP50 exceeds the threshold.

Usage:
    python src/evaluate.py
    python src/evaluate.py --model runs/detect/train/weights/best.pt --threshold 0.75
    python src/evaluate.py --model waste-classification/yolov8n-run/weights/best.pt --threshold 0.5
"""

import os
import sys
import json
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model and apply promotion gate."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="waste-classification/yolov8n-run/weights/best.pt",
        help="Path to trained model weights (default: waste-classification/yolov8n-run/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="src/data.yaml",
        help="Path to dataset config (default: src/data.yaml)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum mAP50 to promote the model (default: 0.75)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Path to write evaluation results (default: metrics.json)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate validation plots (confusion matrix, PR curves)",
    )
    return parser.parse_args()


def evaluate(args):
    """Evaluate model and determine if it should be promoted."""

    model_path = Path(args.model)

    # ============================================================
    # STEP 1: Verify model file exists
    # ============================================================
    if not model_path.exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Run training first: python src/train.py")
        sys.exit(1)

    print("=" * 60)
    print("Model Evaluation Gate")
    print("=" * 60)
    print(f"Model:     {args.model}")
    print(f"Data:      {args.data}")
    print(f"Threshold: mAP50 > {args.threshold}")
    print("=" * 60)

    # ============================================================
    # STEP 2: Load model and run validation
    # ============================================================
    print(f"\nLoading model: {args.model}")
    model = YOLO(str(model_path))

    print("Running validation...")
    metrics = model.val(
        data=args.data,
        plots=args.plots,
        verbose=True,
    )

    # ============================================================
    # STEP 3: Extract metrics
    # ============================================================
    mAP50 = float(metrics.box.map50)
    mAP50_95 = float(metrics.box.map)

    # Per-class metrics
    per_class = {}
    try:
        names = metrics.names  # {0: 'biological', 1: 'cardboard', ...}
        for i, name in names.items():
            per_class[name] = {
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50-95": float(metrics.box.ap[i]),
            }
    except Exception as e:
        print(f"Warning: could not extract per-class metrics: {e}")

    # ============================================================
    # STEP 4: Apply promotion gate
    # ============================================================
    promote = mAP50 > args.threshold

    result = {
        "model": args.model,
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "threshold": args.threshold,
        "promote": promote,
        "per_class": per_class,
    }

    # ============================================================
    # STEP 5: Write results to file
    # ============================================================
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # ============================================================
    # STEP 6: Report
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"mAP50:    {mAP50:.4f}")
    print(f"mAP50-95: {mAP50_95:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Promote:  {promote}")

    if per_class:
        print("\nPer-class mAP50:")
        for name, m in per_class.items():
            print(f"  {name:12s} {m['mAP50']:.4f}")

    print(f"\nResults written to: {args.output}")

    # ============================================================
    # STEP 7: Exit with appropriate code
    # ============================================================
    if not promote:
        print("\n" + "=" * 60)
        print(f"MODEL FAILED PROMOTION GATE")
        print(f"mAP50 ({mAP50:.4f}) is below threshold ({args.threshold})")
        print("Deployment BLOCKED.")
        print("=" * 60)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MODEL PASSED PROMOTION GATE")
    print("Ready for deployment.")
    print("=" * 60)
    return result


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()