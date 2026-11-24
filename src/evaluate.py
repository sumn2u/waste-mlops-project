"""
Evaluation gate for YOLO waste classification model.
Only promotes the model if mAP50 exceeds the threshold.
Logs eval metrics/artifacts to MLflow (same experiment as train.py).

Usage:
    python src/evaluate.py
    python src/evaluate.py --model runs/detect/waste-classification/yolov8n-run/weights/best.pt --threshold 0.5
    python src/evaluate.py --model waste-classification/yolov8n-run/weights/best.pt --threshold 0.5 --experiment garbage-detector
    python src/evaluate.py --no-mlflow     # offline / smoke test
"""

import os
import sys
import json
import argparse
from pathlib import Path

# ============================================================
# MLFLOW FIX: set the SQLite tracking URI BEFORE importing Ultralytics
# so eval runs land in the same store as training/registration.
# ============================================================
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

from ultralytics import YOLO
import mlflow


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
    # ----- NEW -----
    parser.add_argument(
        "--experiment",
        type=str,
        default="garbage-detector",
        help="MLflow experiment name (must match train.py / register_model.py)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging (offline / smoke test).",
    )
    return parser.parse_args()


def _log_to_mlflow(args, metrics, result, per_class):
    """Log eval metrics + artifacts to the active MLflow run."""
    mAP50 = result["mAP50"]
    mAP50_95 = result["mAP50_95"]
    promote = result["promote"]

    # Parameters (also logged at start of the run below, but harmless here)
    mlflow.log_params({
        "model": args.model,
        "data": args.data,
        "threshold": args.threshold,
        "output": args.output,
    })

    # Scalars — use `eval/` prefix so they don't collide with train metrics
    mlflow.log_metrics({
        "eval/mAP50": mAP50,
        "eval/mAP50-95": mAP50_95,
        "eval/threshold": float(args.threshold),
        "eval/promote": 1.0 if promote else 0.0,
    })

    # ============================================================
    # NEW: Standard aggregate detection metrics.
    #
    # Ultralytics returns precision/recall as *per-class arrays*.
    # `.mean()` gives the standard summary numbers that every
    # detection paper/report quotes. F1 is derived from them.
    # These are the missing "standard" scalars for an eval gate.
    # ============================================================
    try:
        precision = float(metrics.box.p.mean())
        recall = float(metrics.box.r.mean())
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mlflow.log_metrics({
            "eval/precision": precision,
            "eval/recall": recall,
            "eval/f1": f1,
        })
    except Exception as e:
        print(f"(skipping aggregate precision/recall/f1: {e})")

    # Number of validation images — useful context when comparing runs
    try:
        n_val = len(metrics.data.val)
        mlflow.log_metric("eval/num_val_images", float(n_val))
    except Exception as e:
        print(f"(skipping num_val_images: {e})")

    # Per-class — mirror train.py naming so charts align
    for name, m in per_class.items():
        mlflow.log_metric(f"per_class/{name}/mAP50", m["mAP50"])
        mlflow.log_metric(f"per_class/{name}/mAP50-95", m["mAP50-95"])
        mlflow.log_metric(f"per_class/{name}/precision", m["precision"])
        mlflow.log_metric(f"per_class/{name}/recall", m["recall"])

    # Tags so runs are filterable in the UI
    mlflow.set_tag("stage", "evaluate")
    mlflow.set_tag("promotion_passed", str(promote))
    mlflow.set_tag("promotion_threshold", str(args.threshold))
    mlflow.set_tag("source_weights", str(Path(args.model).resolve()))

    # metrics.json artifact
    if Path(args.output).exists():
        mlflow.log_artifact(args.output, artifact_path="eval")

    # Plots artifact (only if --plots wrote them)
    if args.plots:
        save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else None
        if save_dir and save_dir.exists():
            for fname in [
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
                "PR_curve.png",
                "F1_curve.png",
                "P_curve.png",
                "R_curve.png",
            ]:
                f = save_dir / fname
                if f.exists():
                    mlflow.log_artifact(str(f), artifact_path="eval/plots")


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
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Threshold:  mAP50 > {args.threshold}")
    print(f"Experiment: {args.experiment}" + (" (MLflow disabled)" if args.no_mlflow else ""))
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
    #
    # NOTE: we also pull mean precision/recall/F1 here so they end up
    # in metrics.json (the artifact) — the standard summary block.
    # ============================================================
    mAP50 = float(metrics.box.map50)
    mAP50_95 = float(metrics.box.map)

    # Standard aggregate detection metrics
    precision_mean = float(metrics.box.p.mean())
    recall_mean = float(metrics.box.r.mean())
    f1_mean = (
        2 * precision_mean * recall_mean / (precision_mean + recall_mean)
        if (precision_mean + recall_mean) > 0
        else 0.0
    )

    per_class = {}
    try:
        names = metrics.names
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
        "precision": precision_mean,
        "recall": recall_mean,
        "f1": f1_mean,
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
    # STEP 6: Log to MLflow (if enabled)
    # ============================================================
    if not args.no_mlflow:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=f"eval-{model_path.parent.parent.name}") as run:
            _log_to_mlflow(args, metrics, result, per_class)
            print(f"\nMLflow run ID: {run.info.run_id}")

    # ============================================================
    # STEP 7: Report
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"mAP50:     {mAP50:.4f}")
    print(f"mAP50-95:  {mAP50_95:.4f}")
    print(f"Precision: {precision_mean:.4f}")
    print(f"Recall:    {recall_mean:.4f}")
    print(f"F1:        {f1_mean:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Promote:   {promote}")

    if per_class:
        print("\nPer-class mAP50:")
        for name, m in per_class.items():
            print(f"  {name:12s} {m['mAP50']:.4f}")

    print(f"\nResults written to: {args.output}")

    # ============================================================
    # STEP 8: Exit with appropriate code
    # ============================================================
    if not promote:
        print("\n" + "=" * 60)
        print("MODEL FAILED PROMOTION GATE")
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