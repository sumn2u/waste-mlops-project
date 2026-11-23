"""
Train YOLO model on waste classification dataset.
MLflow integration logs all metrics, parameters, and artifacts automatically.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch 16 --model yolov8s.pt
    python src/train.py --experiment garbage-detector

Smoke test (fast verification):
    python src/train.py --epochs 3 --imgsz 320 --batch 8 --fraction 0.1 --no-mlflow
"""

import os
import sys
import json
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

# We import mlflow explicitly so we can attach custom logs to the SAME
# run that Ultralytics creates. Ultralytics uses the *active* MLflow
# run if one exists (Ultralytics >= 8.3.x), so wrapping model.train()
# in `with mlflow.start_run(...)` puts everything in one run.
import mlflow


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
        help="Ultralytics project dir name (default: waste-classification)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n-run",
        help="Run name (default: yolov8n-run)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="garbage-detector",
        help="MLflow experiment name (default: garbage-detector). "
             "Use the SAME name in evaluate.py and register_model.py so all "
             "runs land in one experiment and can be compared.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    return parser.parse_args()


def enable_mlflow(experiment: str):
    """
    Enable Ultralytics' native MLflow integration and pin the experiment.

    Once enabled, YOLO automatically logs:
      - Parameters: epochs, batch size, learning rate, model name
      - Metrics: mAP50, mAP50-95, precision, recall per epoch,
                 per-class mAP, box/cls/dfl losses
      - Artifacts: best.pt, last.pt, results.csv, results.png,
                   confusion_matrix.png, PR_curve.png, F1_curve.png,
                   labels.jpg, args.yaml

    The tracking URI is set via the MLFLOW_TRACKING_URI environment
    variable at the top of this file.
    """
    settings.update({"mlflow": True})
    # Pin the experiment so train/eval/register all go to the same place.
    mlflow.set_experiment(experiment)
    print("MLflow tracking enabled for Ultralytics YOLO")
    print(f"Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    print(f"Experiment:   {experiment}")


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


def _dataset_stats(data_yaml: str) -> dict:
    """Return dataset composition for MLflow logging."""
    import yaml
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = Path(data_yaml).parent / root

    train_dir = root / cfg.get("train", "images/train")
    val_dir = root / cfg.get("val", "images/val")

    names = cfg.get("names", [])
    # `names` in a YOLO data.yaml is usually a list; length is the class count.
    num_classes = len(names) if isinstance(names, (list, tuple, dict)) else 0

    return {
        "dataset/train_images": len(list(train_dir.glob("*"))) if train_dir.exists() else 0,
        "dataset/val_images": len(list(val_dir.glob("*"))) if val_dir.exists() else 0,
        "dataset/num_classes": num_classes,
    }


def log_custom_metrics(results, data_yaml: str, save_dir: Path | None):
    """
    Add metrics/artifacts that Ultralytics does NOT log automatically.

    Ultralytics already logs per-epoch mAP, losses, and the standard plots.
    This adds:
      - dataset composition scalars
      - per-class mAP50 / mAP50-95 as individually chartable scalars
      - a small train_summary.json artifact
      - the key plots re-logged under `plots/` for easy access in the UI
    """
    # --- Dataset composition -------------------------------------
    try:
        mlflow.log_metrics(_dataset_stats(data_yaml))
    except Exception as e:
        print(f"(skipping dataset metrics: {e})")

    # --- Final per-class mAP as named scalar metrics -------------
    # Ultralytics logs these only inside results.csv; logging them as
    # scalars lets you chart e.g. "per_class/trash/mAP50 over runs".
    try:
        names = results.names            # {0: 'biological', 1: 'cardboard', ...}
        maps = results.box.maps          # per-class mAP50-95
        map50s = results.box.ap50        # per-class mAP50
        per_class = {}
        for idx, class_name in names.items():
            m50 = float(map50s[idx])
            m5095 = float(maps[idx])
            mlflow.log_metric(f"per_class/{class_name}/mAP50", m50)
            mlflow.log_metric(f"per_class/{class_name}/mAP50-95", m5095)
            per_class[class_name] = {"mAP50": m50, "mAP50-95": m5095}
    except Exception as e:
        print(f"(skipping per-class metrics: {e})")
        per_class = {}

    # --- Summary JSON artifact -----------------------------------
    try:
        summary = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "num_classes": len(results.names),
            "per_class": per_class,
        }
        summary_path = Path("train_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(summary_path), artifact_path="summary")
    except Exception as e:
        print(f"(skipping summary artifact: {e})")

    # --- Re-log key plots under `plots/` -------------------------
    if save_dir and save_dir.exists():
        for fname in [
            "results.csv",
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "labels.jpg",
            "args.yaml",
        ]:
            fpath = save_dir / fname
            if fpath.exists():
                try:
                    mlflow.log_artifact(str(fpath), artifact_path="plots")
                except Exception as e:
                    print(f"(could not log {fname}: {e})")


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLO Training for Waste Classification")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image:      {args.imgsz}")
    print(f"Batch:      {args.batch}")
    print(f"Fraction:   {args.fraction}")
    print(f"Project:    {args.project}")
    print(f"Run:        {args.name}")
    print(f"Experiment: {args.experiment}")
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
        enable_mlflow(args.experiment)
    else:
        print("MLflow tracking disabled")

    # ============================================================
    # STEP 3: Load a pretrained model
    # ============================================================
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)

    # ============================================================
    # STEP 4: Train
    #
    # We open our own MLflow run so the custom logging below lands in
    # the SAME run as Ultralytics' automatic logging. On Ultralytics
    # >= 8.3, model.train() attaches to the active run instead of
    # spawning a new one.
    #
    # If --no-mlflow was passed, we skip the context manager and train
    # directly (no MLflow needed at all).
    # ============================================================
    if args.no_mlflow:
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
        run_id = None
    else:
        with mlflow.start_run(run_name=args.name) as run:
            run_id = run.info.run_id

            # Log the training config up front so it's visible even if
            # training crashes partway through.
            mlflow.log_params({
                "model": args.model,
                "data": args.data,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "fraction": args.fraction,
                "project": args.project,
                "run_name": args.name,
            })
            mlflow.set_tag("stage", "train")
            mlflow.set_tag("model_family", Path(args.model).stem)

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

            save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None

            # Add the custom metrics/artifacts on top of Ultralytics' logs.
            log_custom_metrics(results, args.data, save_dir)

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
        print(f"Save dir:   {save_dir}")
        print(f"Best model: {best_pt}")
        print(f"Last model: {last_pt}")

    try:
        print(f"mAP50:      {results.box.map50:.4f}")
        print(f"mAP50-95:   {results.box.map:.4f}")
    except Exception:
        pass

    if run_id:
        print("\nMLflow:")
        print(f"  Run ID: {run_id}")
        print(f"  UI:     mlflow ui --backend-store-uri {os.environ['MLFLOW_TRACKING_URI']}")
        print(f"  Open:   http://127.0.0.1:5000/#/experiments")

    print("\nNext steps:")
    print("  1. python src/evaluate.py")
    print("  2. python src/register_model.py")
    print("  3. yolo export model=runs/detect/waste-classification/yolov8n-run/weights/best.pt format=tfjs")


if __name__ == "__main__":
    main()