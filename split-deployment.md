
# Waste Classification MLOps Pipeline — Split Deployment Guide (Free / Local MLflow)

An end-to-end MLOps workflow for YOLOv8-based waste detection, split across **Kaggle (training)**, **GitHub (orchestration)**, and **Hugging Face (deployment and monitoring)** — with MLflow running locally and no paid services required.

---

## Architecture

```text
┌────────────────────┐  clone / pull  ┌──────────────────────┐
│ GitHub             │ ─────────────▶ │ Kaggle Notebook      │
│ Code + CI          │ ◀───────────── │ GPU training         │
└─────────┬──────────┘  Kaggle API    └──────────┬───────────┘
          │                                      │
          │                                      │ MLflow logging
          │                                      ▼
          │                             ┌─────────────────────┐
          │                             │ Kaggle working dir  │
          │                             │ mlruns/             │
          │                             └──────────┬──────────┘
          │                                        │
          │                              Package experiment data
          │                                        ▼
          │                             ┌─────────────────────┐
          │                             │ mlflow_artifacts.zip│
          │                             └──────────┬──────────┘
          │                                        │
          │                             Save Notebook Version
          │                                        ▼
          │                             ┌─────────────────────┐
          │                             │ Kaggle Saved Output │
          │                             │ Experiment Archive  │
          │                             └──────────┬──────────┘
          │                                        │
          │                             Download when needed
          ▼                                        ▼
┌────────────────────────────────────────────────────────────┐
│ Local machine                                              │
│ MLflow UI + model evaluation / upload                      │
└───────────────────────────────────────┬────────────────────┘
                                        │ upload model weights
                                        ▼
                             ┌──────────────────────────┐
                             │ Hugging Face Space       │
                             │ Gradio app (CPU Basic)   │
                             └──────────────────────────┘
```

---

## Repository Layout

```text
waste-mlops-project/
├── .github/workflows/       # CI: restart Space and drift monitoring
├── app/                     # Gradio app for Hugging Face Space
├── data/                    # Raw + processed data (DVC-tracked)
├── gx/                      # Great Expectations configuration
├── src/                     # Prepare, train, evaluate, upload, monitor
├── requirements.txt
├── data.dvc
└── README.md
```

---

## Prerequisites

- Python 3.11 (local machine)
- Kaggle account and API credentials
- Hugging Face account and write-access token
- GitHub repository with Actions enabled (public recommended if relying on free GitHub-hosted runner allowances)
- Optional: DVC remote (Google Drive, S3, or another supported storage provider) for dataset persistence

---

## Setup Steps

### 1. MLflow — Local and Free

This workflow uses MLflow's local file-based tracking. No hosted MLflow server, Docker, or cloud database is required.

- During training, MLflow logs experiments to `mlruns/` in the Kaggle working directory, unless another tracking backend is explicitly configured.
- After training, package the MLflow tracking directory into `mlflow_artifacts.zip`.
- Save a Kaggle Notebook version so the archive is retained in the notebook's saved Output.
- Download the archived experiments whenever needed from the saved Notebook Output.
- Restore the archive locally and run the MLflow UI to inspect experiments.

Run the MLflow UI from your local project root:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to inspect the experiments.

> **Important:** Kaggle's active working directory is temporary. Saving a notebook version with the packaged experiment archive allows you to retain the archive as a notebook output for later retrieval. The output is associated with the saved notebook version, not automatically updated by every interactive session.

### 2. Configure GitHub Secrets

In your GitHub repository, go to **Settings → Secrets and variables → Actions** and add:

- `HF_TOKEN` — Hugging Face write token
- `KAGGLE_USERNAME` (optional) — for triggering Kaggle notebooks
- `KAGGLE_KEY` (optional) — for triggering Kaggle notebooks

`MLFLOW_TRACKING_URI`, `DATABASE_URL`, and `ARTIFACT_URI` are not required for this local MLflow setup.

### 3. Train on Kaggle (GPU)

In a Kaggle Notebook with GPU and Internet access enabled, run:

```python
!git clone https://github.com/sumn2u/waste-mlops-project.git
%cd waste-mlops-project
!pip install -r requirements.txt
!kaggle datasets download -d sumn2u/garbage-classification-v2-bbox-384
!unzip -o garbage-classification-v2-bbox-384.zip -d data/
!python src/prepare_data.py --raw data --out data/processed
!python src/validate_data.py
!python src/train.py --epochs 100 --imgsz 640 --batch 16 --fraction 1.0
```

Do **not** set `MLFLOW_TRACKING_URI` unless you intentionally want to use a different tracking backend. Confirm the tracking configuration in your project code and MLflow version.

#### Save MLflow Experiments to Kaggle Output

After training, package the MLflow tracking directory:

```python
import os
import shutil

mlflow_dir = "mlruns"
archive_path = "/kaggle/working/mlflow_artifacts"

if os.path.exists(mlflow_dir):
    shutil.make_archive(
        archive_path,
        "zip",
        root_dir=mlflow_dir
    )
    print("MLflow archive created:", archive_path + ".zip")
else:
    print("MLflow tracking directory not found:", mlflow_dir)
```

The generated archive will be:

```text
/kaggle/working/mlflow_artifacts.zip
```

The archive preserves the MLflow tracking directory structure, including experiment metadata, run metadata, metrics, parameters, tags, and locally stored artifacts.

#### Persist the Experiments

1. Run the training and archive cells in the Kaggle Notebook.
2. Click **Save Version**.
3. Choose the appropriate notebook version save/run option.
4. Wait for the saved notebook version to complete.
5. Open the saved version and access its Output section.
6. Download `mlflow_artifacts.zip` whenever needed.

The saved Kaggle Notebook Output serves as the experiment archive, allowing you to retrieve the MLflow tracking data after the active notebook session has ended.

> **Note:** The archive must actually be generated during the saved notebook version's execution. Saving a version does not automatically preserve files created only in an unrelated interactive session.

### 4. Restore MLflow Experiments Locally

Download `mlflow_artifacts.zip` from the saved Kaggle Notebook Output.

Extract it into your local project root:

```bash
unzip mlflow_artifacts.zip
```

Ensure the `mlruns/` directory is restored in the project root.

Launch MLflow:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

You can now inspect:

- Training metrics
- Hyperparameters
- Experiment runs
- Run comparisons
- Logged artifacts

No hosted MLflow service or paid tracking infrastructure is required.

### 5. Evaluate the Model

Run the evaluation script from your project root. Replace the model path with the path to the trained weights if needed:

```bash
python src/evaluate.py \
  --model waste-classification/yolov8n-run/weights/best.pt \
  --threshold 0.75
```

### 6. Publish the Model to Hugging Face

First, copy the trained model into the app's expected location, then upload it:

```bash
python src/copy_model.py

python src/upload_model.py \
  --repo <your-username>/<your-space-name> \
  --repo-type space \
  --token "$HF_TOKEN"
```

Replace `<your-username>/<your-space-name>` with your Space repository ID (for example, `iamsuman/waste-mlops`).

Ensure `HF_TOKEN` is available in your shell environment and has write permissions for the Space.

### 7. Hugging Face Space — Free CPU Basic

Create a Hugging Face Space using **CPU Basic (free)** hardware.

- The `@spaces.GPU` decorator in `app.py` may need to be removed or adjusted if the app is intended to run strictly on CPU. Check the Space's runtime requirements.
- YOLOv8n inference will generally be slower on CPU than on GPU, but the app can still run if its dependencies and inference code support CPU.
- When `app/best.pt` is pushed to GitHub, `.github/workflows/restart-space.yml` can restart the Space to load the updated weights.

### 8. Scheduled Drift Monitoring

The `.github/workflows/monitor.yml` workflow is scheduled to run weekly. It executes:

```bash
python src/generate_features.py
python src/monitor.py
```

The workflow uploads `drift_report.html` as a GitHub Actions artifact.

The repository may be public to make it easier to work within applicable GitHub-hosted Actions free usage limits; actual limits and eligibility depend on GitHub's current policies.

### 9. Optional: Automatic Retraining on Drift

Extend `src/monitor.py` or the workflow to trigger `kaggle kernels_push` when drift exceeds a configured threshold.

This requires `KAGGLE_USERNAME` and `KAGGLE_KEY` in GitHub Actions secrets, plus a configured Kaggle notebook and appropriate API permissions.

---

## Configuration Reference

| Variable | Where | Purpose |
|---|---|---|
| `HF_TOKEN` | GitHub Actions secret / local environment | Upload model and restart Space |
| `KAGGLE_USERNAME` | GitHub Actions secret (optional) | Authenticate Kaggle API |
| `KAGGLE_KEY` | GitHub Actions secret (optional) | Authenticate Kaggle API |
| `KAGGLE_DATASET` | Kaggle | Dataset identifier, e.g. `sumn2u/garbage-classification-v2-bbox-384` |

`MLFLOW_TRACKING_URI`, `DATABASE_URL`, and `ARTIFACT_URI` are not used in this free/local configuration unless explicitly configured by your code.

---

## What Runs Where (Quick Matrix)

| Component | Kaggle | GitHub Actions | Hugging Face Space | Local |
|---|:---:|:---:|:---:|:---:|
| Data preparation / validation | ✅ | | | |
| Model training | ✅ | | | |
| MLflow logging | ✅ (local files) | | | |
| MLflow archive packaging | ✅ | | | |
| MLflow archive persistence | ✅ (saved Notebook Output) | | | |
| MLflow UI | | | | ✅ |
| Evaluation | ✅ / local, as configured | | | |
| Model upload | | | | ✅ |
| App serving | | | ✅ (CPU Basic) | |
| Drift report | | ✅ | | |
| Retraining trigger | | ✅ (optional) | | |

---

## GitHub Workflow Files

### `.github/workflows/restart-space.yml`

```yaml
name: Restart HF Space

on:
  push:
    paths:
      - "app/best.pt"
  workflow_dispatch:

jobs:
  restart:
    runs-on: ubuntu-latest
    steps:
      - name: Restart Space
        run: |
          pip install huggingface_hub
          python - <<'PY'
          import os
          from huggingface_hub import HfApi

          HfApi(token=os.environ["HF_TOKEN"]).restart_space(
              repo_id=os.environ["SPACE_REPO"],
              factory_reboot=True,
          )
          PY
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          SPACE_REPO: <your-username>/<your-space-name>
```

Replace `SPACE_REPO` with your Hugging Face Space repository ID. You can also define it as a GitHub Actions variable and reference it through `env`.

### `.github/workflows/monitor.yml`

```yaml
name: Drift Monitor

on:
  schedule:
    - cron: "0 6 * * 1" # Every Monday at 06:00 UTC
  workflow_dispatch:

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Generate features
        run: python src/generate_features.py

      - name: Run drift monitoring
        run: python src/monitor.py

      - name: Upload drift report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: drift-report
          path: drift_report.html
          if-no-files-found: warn
```

---

## Notes & Gotchas

- **Kaggle persistence:** Package `mlruns/` into `mlflow_artifacts.zip` and save a Kaggle Notebook version. Retrieve the archive from the saved Notebook Output whenever needed.
- **MLflow archive:** Retain the complete tracking directory structure when packaging and restoring experiment data.
- **Local MLflow:** Does not provide remote live viewing in this setup; it is free and does not require a hosted account.
- **CPU inference:** Hugging Face CPU Basic is free under the applicable current Space plan, but performance may be slower than GPU inference.
- **GitHub Actions usage:** Public repositories may have different runner allowances and policies; verify current GitHub limits.
- **DVC remote:** Configure a remote (e.g., Google Drive or S3) to avoid downloading the dataset again every Kaggle session.
- **No ZeroGPU:** This configuration does not use ZeroGPU, so it does not rely on per-user GPU quota.

---

## License

Educational / demonstration use. See the original project for credits.
