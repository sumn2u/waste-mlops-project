"""
Register a trained YOLO waste-classification model in the MLflow Model Registry.

This script assumes training has already been run (src/train.py), so a
best.pt exists under runs/detect/.../weights/best.pt and the tracking
URI is the SQLite store used during training.

Usage:
    python src/register_model.py
    python src/register_model.py --weights runs/detect/waste-classification/yolov8n-run/weights/best.pt
    python src/register_model.py --model-name garbage-detector --experiment garbage-detector

Smoke test / dry-run (no registry write):
    python src/register_model.py --dry-run
"""

import os
import argparse
from pathlib import Path

# ------------------------------------------------------------
# Anchor BOTH the tracking DB and the artifact root to the project
# root, regardless of the shell's CWD. This is what makes MLmodel
# land at <project>/mlruns/<exp>/<run>/artifacts/model/MLmodel
# instead of being scattered into wherever CWD happened to be.
# ------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/../ = project root

os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{_PROJECT_ROOT}/mlflow.db",
)
os.environ.setdefault(
    "MLFLOW_DEFAULT_ARTIFACT_ROOT",
    str(_PROJECT_ROOT / "mlruns"),
)

import mlflow
import mlflow.pyfunc
from ultralytics import YOLO


class YOLOWrapper(mlflow.pyfunc.PythonModel):
    """Thin pyfunc wrapper so a YOLO .pt can be served/loaded via MLflow."""

    def load_context(self, context):
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input):
        # model_input is expected to be a list of image paths (or a single path)
        return self.model(model_input)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register a trained YOLO model in the MLflow Model Registry."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/waste-classification/yolov8n-run/weights/best.pt",
        help="Path to trained weights (default: runs/detect/waste-classification/yolov8n-run/weights/best.pt)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="garbage-detector",
        help="MLflow experiment name (default: garbage-detector)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="garbage-detector",
        help="Registered model name (default: garbage-detector)",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="model",
        help="Artifact path within the run (default: model)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate weights and log the model artifact, but skip registration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # STEP 1: Verify weights exist
    # ------------------------------------------------------------
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}. "
            f"Run `python src/train.py` first."
        )
    print(f"Weights: {weights_path.resolve()}")

    # ------------------------------------------------------------
    # STEP 2: Point MLflow at the same store + artifact root used during training
    # ------------------------------------------------------------
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    artifact_root = os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    print(f"Tracking URI:    {tracking_uri}")
    print(f"Artifact root:   {artifact_root}")
    print(f"Experiment:      {args.experiment}")

    # ------------------------------------------------------------
    # STEP 3: Log + register the model
    # ------------------------------------------------------------
    with mlflow.start_run(run_name=f"register-{args.model_name}") as run:
        mlflow.log_param("weights_source", str(weights_path.resolve()))
        mlflow.log_param("base_model", "yolov8n")  # adjust if you change --model

        kwargs = dict(
            name=args.artifact_path,
            python_model=YOLOWrapper(),
            artifacts={"weights": str(weights_path.resolve())},
            # Pin runtime deps so `mlflow models serve` can rebuild the env.
            pip_requirements=["ultralytics>=8.4", "torch", "pandas"],
        )
        if not args.dry_run:
            kwargs["registered_model_name"] = args.model_name

        model_info = mlflow.pyfunc.log_model(**kwargs)

        print(f"Run ID:    {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")

        # ------------------------------------------------------------
        # STEP 4: Verify the MLmodel file was actually written
        # ------------------------------------------------------------
        run_artifact_dir = (
            _PROJECT_ROOT
            / "mlruns"
            / str(run.info.experiment_id)
            / run.info.run_id
            / "artifacts"
        )
        mlmodel_path = run_artifact_dir / args.artifact_path / "MLmodel"
        # The model_info.model_uri may carry its own artifact path; also check it.
        if mlmodel_path.exists():
            print(f"MLmodel:   {mlmodel_path}")
        else:
            # Fall back to scanning — protects against custom artifact roots.
            candidates = list((_PROJECT_ROOT / "mlruns").rglob("MLmodel"))
            if candidates:
                print(f"MLmodel:   {candidates[-1]}  (found via scan)")
            else:
                print(
                    "WARNING: MLmodel file not found under "
                    f"{_PROJECT_ROOT / 'mlruns'}. "
                    "Check MLFLOW_DEFAULT_ARTIFACT_ROOT."
                )

        if not args.dry_run:
            print(f"Registered model name: {args.model_name}")
            # Optionally promote to a stage / alias:
            # client = mlflow.tracking.MlflowClient()
            # client.set_registered_model_alias(
            #     args.model_name, "staging", model_info.registered_model_version
            # )
        else:
            print("Dry run: model logged but NOT registered.")


if __name__ == "__main__":
    main()