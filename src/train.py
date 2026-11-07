"""
Train YOLO model on waste classification dataset.
MLflow integration logs all metrics, parameters, and artifacts automatically.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch 16 --model yolov8s.pt

Smoke test (fast verification):
    python src/train.py --epochs 3 --imgsz 320 --batch 8 --fraction 0.1 --no-mlflow
"""

import os
import sys
import argparse
from pathlib import Path

# ============================================================
# MLFLOW FIX: Set a SQLite tracking URI BEFORE importing Ultralytics.
#
# MLflow's filesystem backend (./mlruns) is in maintenance mode
# and raises an exception by default. SQLite is the recommended
# replacement for local development.
# ============================================================
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Ultralytics has native MLflow integration — no manual logging needed
from ultralytics import settings, YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO on waste classification dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO model (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="src/data.yaml",
        help="Path to dataset config (default: src/data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (default: 1.0). Use 0.1 for smoke tests.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="waste-classification",
        help="MLflow experiment / project name (default: waste-classification)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n-run",
        help="Run name (default: yolov8n-run)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    return parser.parse_args()


def enable_mlflow():
    """
    Enable Ultralytics' native MLflow integration.

    Once enabled, YOLO automatically logs:
      - Parameters: epochs, batch size, learning rate, model name
      - Metrics: mAP50, mAP50-95, precision, recall per epoch
      - Artifacts: best.pt weights, config files, training plots

    The tracking URI is set via the MLFLOW_TRACKING_URI environment
    variable at the top of this file.
    """
    settings.update({"mlflow": True})
    print("MLflow tracking enabled for Ultralytics YOLO")
    print(f"Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")


def check_dataset(data_yaml: str) -> bool:
    """Verify the dataset config exists and points to real files."""
    data_path = Path(data_yaml)

    if not data_path.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        return False

    # Read the yaml to find the dataset root
    import yaml
    with open(data_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve the path relative to the yaml file
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = data_path.parent / root

    train_dir = root / cfg.get("train", "images/train")
    val_dir = root / cfg.get("val", "images/val")

    if not train_dir.exists():
        print(f"ERROR: train directory not found: {train_dir}")
        return False

    if not val_dir.exists():
        print(f"ERROR: val directory not found: {val_dir}")
        return False

    n_train = len(list(train_dir.glob("*")))
    n_val = len(list(val_dir.glob("*")))

    print(f"Dataset config: {data_yaml}")
    print(f"  Root:  {root}")
    print(f"  Train: {n_train} files")
    print(f"  Val:   {n_val} files")

    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLO Training for Waste Classification")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Epochs:   {args.epochs}")
    print(f"Image:    {args.imgsz}")
    print(f"Batch:    {args.batch}")
    print(f"Fraction: {args.fraction}")
    print(f"Project:  {args.project}")
    print(f"Run:      {args.name}")
    print("=" * 60)

    # ============================================================
    # STEP 1: Verify the dataset config
    # ============================================================
    if not check_dataset(args.data):
        print("Dataset check failed. Run prepare_data.py first.")
        sys.exit(1)

    # ============================================================
    # STEP 2: Enable MLflow tracking
    # ============================================================
    if not args.no_mlflow:
        enable_mlflow()
    else:
        print("MLflow tracking disabled")

    # ============================================================
    # STEP 3: Load a pretrained model
    # ============================================================
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)

    # ============================================================
    # STEP 4: Train
    # ============================================================
    print("\nStarting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # ============================================================
    # STEP 5: Report results
    # ============================================================
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if save_dir:
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"
        print(f"Save dir:  {save_dir}")
        print(f"Best model: {best_pt}")
        print(f"Last model: {last_pt}")

    try:
        print(f"mAP50:     {results.box.map50:.4f}")
        print(f"mAP50-95:  {results.box.map:.4f}")
    except Exception:
        pass

    print("\nNext steps:")
    print("  1. python src/evaluate.py")
    print("  2. yolo export model=runs/detect/train/weights/best.pt format=tfjs")


if __name__ == "__main__":
    main()