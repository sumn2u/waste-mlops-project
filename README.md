<div align="center">

# Waste Classification MLOps Pipeline

**An end-to-end MLOps workflow for YOLOv8-based waste detection — from dataset preparation and model training to gated deployment on Hugging Face Spaces.**

[![Model](https://img.shields.io/badge/Model-YOLOv8n-8A2BE2)](https://github.com/ultralytics/ultralytics)
[![Interface](https://img.shields.io/badge/UI-Gradio-orange)](https://www.gradio.app/)
[![Deployment](https://img.shields.io/badge/Deployment-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/iamsuman/waste-mlops)
[![Tracking](https://img.shields.io/badge/Experiments-MLflow-blue)](https://mlflow.org/)
[![Data Versioning](https://img.shields.io/badge/Data-DVC-945DD6)](https://dvc.org/)

[**Launch the live application**](https://iamsuman-waste-mlops.hf.space) · [Hugging Face Space](https://huggingface.co/spaces/iamsuman/waste-mlops)

</div>

---

## Contents

- [Overview](#overview)
- [Supported Waste Classes](#supported-waste-classes)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Pipeline Workflow](#pipeline-workflow)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Current Status and Roadmap](#current-status-and-roadmap)
- [Architectural Decisions](#architectural-decisions)
- [License and Acknowledgments](#license-and-acknowledgments)

## Overview

This project implements an end-to-end machine learning operations (MLOps) pipeline for detecting and classifying waste in images using **Ultralytics YOLOv8n**.

The workflow brings together dataset versioning, data preparation, validation, model training, experiment tracking, evaluation, model publishing, and application deployment. The inference application is built with **Gradio** and hosted on **Hugging Face Spaces with ZeroGPU**, which allocates GPU resources on demand.

### Pipeline capabilities

- Version datasets with **DVC**.
- Prepare and validate YOLO-format images and labels.
- Train YOLOv8n and track experiments with **MLflow**.
- Apply an evaluation gate before promoting a model.
- Upload model weights to the Hugging Face Hub using the Hub API.
- Serve image-based predictions through a Gradio application.
- Generate image features for production drift monitoring.
- Plan production data-drift monitoring and automated retraining.

## Supported Waste Classes

The model is configured for seven waste categories:

| Class ID | Category |
|---:|---|
| 0 | Biological |
| 1 | Cardboard |
| 2 | Glass |
| 3 | Metal |
| 4 | Paper |
| 5 | Plastic |
| 6 | Trash |

> Class IDs should remain consistent between the dataset labels, `src/data.yaml`, training, and inference.

## System Architecture

The pipeline is organized into six stages: data management, training, evaluation, model synchronization, deployment, and monitoring.

## End-to-End Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA & VERSION CONTROL                              │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │ Kaggle       │───▶│ DVC          │───▶│ data/processed/              │   │
│  │ Dataset      │    │ (versioning) │    │   images/train, val          │   │
│  │ (BBox 384)   │    │              │    │   labels/train, val          │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘   │
│          │                                      │                           │
│          ▼                                      ▼                           │
│  ┌────────────────────┐              ┌──────────────────────────────┐       │
│  │ prepare_data.py    │              │ validate_data.py             │       │
│  │ Reorganize data    │              │ (Great Expectations 1.23.2)  │       │
│  │ and remap classes  │              │ Schema and range checks      │       │
│  └────────────────────┘              └──────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING & EXPERIMENT TRACKING                           │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ train.py (YOLOv8n)                                                    │  │
│  │ • Kaggle GPU (T4/P100)                                                │  │
│  │ • --fraction for smoke tests                                          │  │
│  │ • Native Ultralytics–MLflow integration                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                       ┌──────────────┴──────────────┐                       │
│                       ▼                             ▼                       │
│  ┌────────────────────────────┐  ┌──────────────────────────────────────┐   │
│  │ MLflow (SQLite)            │  │ waste-classification/yolov8n-run/    │   │
│  │ mlflow.db                  │  │   weights/best.pt                    │   │
│  │ • Parameters               │  │   weights/last.pt                    │   │
│  │ • Metrics (mAP50)          │  │   results.csv                        │   │
│  │ • Artifacts                │  │                                      │   │
│  └────────────────────────────┘  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION GATE                                │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ evaluate.py                                                           │  │
│  │ • Loads best.pt                                                       │  │
│  │ • Runs model.val(data.yaml)                                           │  │
│  │ • Compares mAP50 against --threshold                                  │  │
│  │ • Writes metrics.json with promote: true/false                        │  │
│  │ • Exits with code 1 below threshold, blocking deployment              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                         ┌────────────┴────────────┐                         │
│                         ▼                         ▼                         │
│                  ┌────────────┐             ┌────────────┐                  │
│                  │ PASS       │             │ FAIL       │                  │
│                  │ (promote)  │             │ (blocked)  │                  │
│                  └─────┬──────┘             └────────────┘                  │
└────────────────────────┼────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL REGISTRY & SYNC                             │
│                                                                             │
│  ┌────────────────────────────┐       ┌──────────────────────────────────┐  │
│  │ copy_model.py              │──────▶│ app/best.pt                      │  │
│  │ • Validate model           │       │ (deployment copy)                │  │
│  │ • Copy model               │       └────────────────┬─────────────────┘  │
│  │ • Check Git tracking       │                        │                    │
│  └────────────────────────────┘                        ▼                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ upload_model.py (huggingface_hub API)                                 │  │
│  │ • Bypasses Git LFS/Xet binary rejection                               │  │
│  │ • Uploads best.pt directly to the Hugging Face Space repository       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Hugging Face Hub — iamsuman/waste-mlops (Space)                       │  │
│  │ • best.pt stored using Xet storage, not Git                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT (HF SPACES)                            │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ GitHub Repository (app/)                                              │  │
│  │ • app.py — Gradio + @spaces.GPU                                       │  │
│  │ • requirements.txt — no Streamlit, no pinned torch                    │  │
│  │ • README.md — sdk: gradio, sdk_version: 5.0.0                         │  │
│  │ • .gitattributes                                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼ git push origin main                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Hugging Face Space (ZeroGPU)                                          │  │
│  │ • Downloads best.pt using hf_hub_download() at startup                │  │
│  │ • @spaces.GPU(duration=60) allocates GPU on demand                    │  │
│  │ • YOLO inference returns an annotated image and detection list        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Live Application                                                      │  │
│  │ https://iamsuman-waste-mlops.hf.space                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING (PLANNED)                                │
│                                                                             │
│  ┌────────────────────────────┐       ┌──────────────────────────────────┐  │
│  │ generate_features.py       │──────▶│ reference_features.csv           │  │
│  │ • Extract image features   │       │ production_features.csv          │  │
│  │ • Verify schemas match     │       └────────────────┬─────────────────┘  │
│  └────────────────────────────┘                        │                    │
│                                                        ▼                    │
│  ┌────────────────────────────┐       ┌──────────────────────────────────┐  │
│  │ Evidently                  │──────▶│ Drift report                     │  │
│  │ (DataDriftPreset)          │       │ drift_report.html                │  │
│  └──────────────┬─────────────┘       └──────────────────────────────────┘  │
│                 ▼                                                           │
│  ┌────────────────────────────┐                                             │
│  │ Trigger retraining         │──────────────▶ Back to train.py             │
│  │ (GitHub API)               │                 (closed loop)               │
│  └────────────────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```


### Stage details

| Stage | Main component | Responsibility |
|---|---|---|
| Data versioning | DVC | Track datasets and pipeline data outside Git |
| Data preparation | `prepare_data.py` | Organize images and labels into YOLO train/validation folders; remap class IDs |
| Data validation | `validate_data.py` | Check YOLO label schema and value ranges with Great Expectations |
| Training | `train.py` | Train YOLOv8n; supports fractional-data smoke tests |
| Experiment tracking | MLflow | Log parameters, metrics, and artifacts to SQLite |
| Evaluation | `evaluate.py` | Compare validation mAP50 with a configured threshold |
| Model synchronization | `copy_model.py`, `upload_model.py` | Validate, copy, and publish model weights |
| Deployment | Gradio + Hugging Face Spaces | Host the prediction interface and allocate GPU on demand |
| Feature generation | `generate_features.py` | Extract image statistics for drift monitoring |
| Monitoring | `monitor.py` | Planned data-drift reporting and retraining trigger |

## Repository Structure

```text
waste-mlops-project/
├── .github/
│   └── workflows/              # CI/CD workflow definitions
├── app/
│   ├── app.py                  # Gradio inference application
│   ├── best.pt                 # Deployment model copy (Git-ignored)
│   ├── Dockerfile              # Optional Docker setup
│   ├── README.md               # Hugging Face Space metadata
│   └── requirements.txt        # Application dependencies
├── data/
│   ├── raw/                    # Original Kaggle dataset (DVC-tracked)
│   └── processed/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── gx/
│   ├── expectations/
│   └── great_expectations.yml  # Great Expectations configuration
├── runs/                       # YOLO validation outputs (ignored)
├── src/
│   ├── copy_model.py           # Copy trained weights to app/
│   ├── data.yaml               # Dataset and class configuration
│   ├── evaluate.py             # Model evaluation and promotion gate
│   ├── generate_features.py    # Extract image features for drift monitoring
│   ├── monitor.py              # Data-drift monitoring
│   ├── prepare_data.py         # Dataset reorganization
│   ├── train.py                # YOLO training with MLflow
│   ├── upload_model.py         # Upload weights to Hugging Face Hub
│   └── validate_data.py        # Dataset validation
├── waste-classification/       # YOLO training artifacts (ignored)
├── .gitignore
├── data.dvc                    # DVC dataset pointer
├── mlflow.db                   # MLflow SQLite backend (ignored)
├── reference_features.csv      # Baseline features for drift monitoring (ignored)
├── production_features.csv     # Production features for drift monitoring (ignored)
├── requirements.txt            # Pipeline dependencies
└── README.md
```

## Technology Stack

| Area | Tool / technology | Version / configuration | Purpose |
|---|---|---|---|
| Dataset | [Kaggle](https://www.kaggle.com/) | `sumn2u/garbage-classification-v2-bbox-384` | Seven-class YOLO bounding-box dataset |
| Data versioning | [DVC](https://dvc.org/) | Latest | Track datasets and pipeline stages outside Git |
| Data preparation | Python, `pathlib`, `shutil` | — | Reorganize data and remap class IDs |
| Data validation | [Great Expectations](https://greatexpectations.io/) | 1.23.2 | Validate YOLO label schemas and ranges |
| Model training | [Ultralytics YOLO](https://www.ultralytics.com/) | 8.3.74 | YOLOv8n object detection |
| Training compute | Kaggle GPU | T4 / P100 | Model training |
| Local fallback | Apple M2 CPU | PyTorch 2.3.0 | Smoke tests |
| Experiment tracking | [MLflow](https://mlflow.org/) | 2.x | Track parameters, metrics, and artifacts |
| Tracking backend | SQLite | `mlflow.db` | Local MLflow backend |
| Evaluation | Ultralytics `model.val` | `evaluate.py` | Gate promotion using mAP50 |
| Model publishing | `huggingface_hub` | `HfApi.upload_file` | Upload model weights to the Hub |
| Application | [Gradio](https://www.gradio.app/) | 5.0.0 | Image upload and prediction interface |
| GPU allocation | `spaces.GPU` | `duration=60` | On-demand ZeroGPU allocation |
| Runtime download | `hf_hub_download` | — | Download model weights at application startup |
| Inference | Ultralytics YOLO, OpenCV, Pillow | — | Object detection and image annotation |
| Feature extraction | OpenCV, pandas | `generate_features.py` | Extract RGB means, brightness, dimensions for drift detection |
| Deployment | [Hugging Face Spaces](https://huggingface.co/spaces) | Gradio SDK / ZeroGPU | Host the inference application |
| Monitoring | [Evidently](https://www.evidentlyai.com/) | 0.4.28 (planned) | Detect production data drift |
| Retraining trigger | GitHub Actions API | Planned | Trigger a retraining workflow |

## Getting Started

### Prerequisites

- Python 3.11
- Git
- DVC
- A Kaggle account and API credentials for dataset access
- A Hugging Face account and write-access token for model publishing

### 1. Clone the repository

Replace the placeholder repository URL with the actual GitHub repository URL.

```bash
git clone https://github.com/sumn2u/waste-mlops-project.git
cd waste-mlops-project
```

### 2. Install pipeline dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and prepare the dataset

```bash
kaggle datasets download -d sumn2u/garbage-classification-v2-bbox-384
unzip garbage-classification-v2-bbox-384.zip -d data/
python3 src/prepare_data.py --raw data --out data/processed
```

### 4. Validate the dataset

```bash
python3 src/validate_data.py
```

### 5. Train the model

Run a small smoke test first to verify that the pipeline executes:

```bash
python3 src/train.py \
  --epochs 3 \
  --imgsz 320 \
  --batch 8 \
  --fraction 0.1 \
  --no-mlflow
```

Run full training when the data and configuration are ready:

```bash
python3 src/train.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --fraction 1.0
```

### 6. Evaluate the trained model

The evaluation script compares validation mAP50 against the configured threshold. A score below the threshold blocks promotion.

```bash
python3 src/evaluate.py \
  --model waste-classification/yolov8n-run/weights/best.pt \
  --threshold 0.75
```

### 7. Generate features for drift monitoring

Extract image statistics from the training and validation folders. This creates the two CSVs that `monitor.py` compares:

```bash
python3 src/generate_features.py
```
This writes:

- `reference_features.csv` — baseline features from `data/processed/images/train/`
- `production_features.csv` — comparison features from `data/processed/images/val/`

Optionally override the source folders:

```bash
python3 src/generate_features.py \
  --reference-dir data/processed/images/train \
  --production-dir data/processed/images/val \
  --max-samples 1000
```

### 8. Publish the model

```bash
python3 src/copy_model.py
python3 src/upload_model.py \
    --repo iamsuman/waste-mlops \
    --repo-type space \
    --token $(hf auth token)
```

Make sure your Hugging Face credentials are configured before uploading.

## Pipeline Workflow

The main pipeline scripts and their outputs are summarized below.

| Step | Component | Output |
|---:|---|---|
| 1 | DVC | `data.dvc` |
| 2 | `prepare_data.py` | `data/processed/` |
| 3 | `validate_data.py` | Dataset validation result |
| 4 | `train.py` | YOLO training weights and run artifacts |
| 5 | MLflow | `mlflow.db` and logged artifacts |
| 6 | `evaluate.py` | `metrics.json` with promotion decision |
| 7 | `copy_model.py` | `app/best.pt` |
| 8 | `upload_model.py` | Model weights in the Hugging Face Space repository |
| 9 | Gradio + Hugging Face Spaces | Live inference application |
| 10 | `generate_features.py` | `reference_features.csv` and `production_features.csv` |
| 11 | `monitor.py` | `drift_report.html` (planned) |

### Evaluation and promotion

`evaluate.py`:

1. Loads the trained `best.pt` checkpoint.
2. Runs Ultralytics validation using `model.val(data.yaml)`.
3. Compares mAP50 against the supplied `--threshold`.
4. Writes `metrics.json` with `promote: true` or `promote: false`.
5. Exits with code `1` when the score is below the threshold, preventing deployment from continuing.

## Deployment

The inference application is hosted on Hugging Face Spaces.

| Setting | Configuration |
|---|---|
| Space | [`iamsuman/waste-mlops`](https://huggingface.co/spaces/iamsuman/waste-mlops) |
| Live application | [iamsuman-waste-mlops.hf.space](https://iamsuman-waste-mlops.hf.space) |
| SDK | Gradio 5.0.0 |
| Hardware | ZeroGPU |
| Application entry point | `app.py` |
| Model source | Hugging Face Space repository |
| Model download | `hf_hub_download()` |

### Inference flow

1. The Space starts the Gradio application.
2. The application downloads `best.pt` using `hf_hub_download()`.
3. The prediction function requests GPU resources with `@spaces.GPU(duration=60)`.
4. YOLO performs inference and returns an annotated image and detection results.

The GPU is allocated for prediction rather than held continuously while the application is idle.

### Application dependencies

The application uses the following packages:

```text
ultralytics
huggingface_hub
pillow
numpy
opencv-python-headless
spaces
```

The application dependencies are maintained separately in `app/requirements.txt`.

## Monitoring

**Status: Planned**

Production data-drift monitoring is planned with Evidently. The monitoring workflow has two parts.

### Feature generation (implemented)

`generate_features.py` extracts seven numerical features from each image:

| Feature | Description |
|---|---|
| `mean_r` | Mean red channel value |
| `mean_g` | Mean green channel value |
| `mean_b` | Mean blue channel value |
| `brightness` | Mean grayscale value |
| `width` | Image width in pixels |
| `height` | Image height in pixels |
| `aspect_ratio` | width / height |

It writes two CSVs with identical schemas:

- `reference_features.csv` — from the training folder (baseline)
- `production_features.csv` — from the validation folder (or real production logs later)

The script verifies that both CSVs have the same columns before exiting, so `monitor.py` cannot silently compare mismatched schemas.

### Drift detection (planned)

`monitor.py` is intended to:

- Load `reference_features.csv` and `production_features.csv`.
- Compare distributions using Evidently's `DataDriftPreset`.
- Use statistical tests such as the Kolmogorov–Smirnov (K-S) test and chi-square test.
- Generate an HTML report named `drift_report.html`.
- Trigger retraining through the GitHub API when drift exceeds a configured threshold.

The intended long-term workflow closes the loop by feeding a retraining trigger back into `train.py`.

## Current Status and Roadmap

| Component | Current state | Planned next step |
|---|---|---|
| Model quality | Smoke test reported mAP50 of approximately 0.03 | Full 100-epoch training; target mAP50 above 0.5 |
| Feature generation | Implemented (`generate_features.py`) | Extend to log real production features from the Gradio app |
| Data-drift monitoring | Script written; not deployed | Evidently report and scheduled monitoring job |
| Automated retraining | Manual | GitHub Actions trigger based on detected drift |
| CI/CD | Manual `git push` | GitHub Actions deployment workflow for Hugging Face |
| Dataset tracking | Local DVC | Configure a remote such as S3 or Google Drive |

> The model-quality figures above describe the smoke-test state and a planned target; they are not a claim of production model performance.

## Architectural Decisions

| Decision | Rationale |
|---|---|
| **DVC + MLflow** | DVC manages dataset versioning; MLflow manages experiment tracking. They address different parts of the workflow. |
| **SQLite for MLflow** | Uses a SQLite backend instead of MLflow's maintenance-mode file-store backend. |
| **Hugging Face Hub API for model uploads** | Publishes model weights through `huggingface_hub` and the Hub's Xet-backed storage rather than committing large binaries through Git. |
| **Gradio for the application** | Aligns the application with the Gradio SDK used by the Hugging Face Spaces deployment. |
| **`@spaces.GPU(duration=60)`** | Requests GPU resources for inference on demand. |
| **Separate `app/` directory** | Keeps deployment code and application dependencies separate from the training pipeline. |
| **Evaluation gate before deployment** | Prevents a model below the configured evaluation threshold from being promoted. |
| **Image statistics for drift monitoring** | RGB means, brightness, and dimensions are cheap to compute, schema-stable, and sensitive to lighting, camera, and resolution shifts that matter for waste detection. |

## License and Acknowledgments

This project is intended for educational and demonstration purposes.

Thanks to the teams and communities behind:

- **Dataset:** [`sumn2u/garbage-classification-v2-bbox-384`](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2-bbox-384) on Kaggle
- **Object detection:** [Ultralytics](https://www.ultralytics.com/)
- **Experiment tracking:** [MLflow](https://mlflow.org/)
- **Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces)
- **Data validation:** [Great Expectations](https://greatexpectations.io/)
- **Data versioning:** [DVC](https://dvc.org/)

---

<div align="center">

**Built for reproducible waste-classification experiments and practical deployment.**

[Live Demo](https://iamsuman-waste-mlops.hf.space) · [Hugging Face Space](https://huggingface.co/spaces/iamsuman/waste-mlops)

</div>
