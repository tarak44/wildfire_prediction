Wildfire Prediction — MLOps‑Grade Project

Overview
This repository packages a wildfire image classifier into a clean, production‑style MLOps structure: modular training and inference, versioned data, experiment tracking, monitoring hooks, and automated tests.

Highlights
- Modular package under `src/` with clear separation of core, data, training, inference, and monitoring
- FastAPI for programmatic inference and Streamlit for demos
- CLI for local inference, batch inference, and evaluation
- MLflow for experiment tracking and optional model registry
- DVC for dataset versioning and pipeline reproducibility
- CI with linting, tests, and security audit

Project structure
- `src/wildfire_mlops/`   Core package
- `api/`                  FastAPI app
- `apps/`                 Streamlit app
- `tests/`                Pytest tests
- `configs/`              Experiment configs
- `data/`                 Dataset (DVC‑tracked, not committed)

Quickstart
1) Install
   `pip install -e .`

2) Run API
   `uvicorn api.main:app --host 0.0.0.0 --port 8000`

3) Run Streamlit
   `streamlit run apps/streamlit_app.py`

4) CLI
   `wildfire-cli --image path/to/image.jpg`
   `wildfire-cli --input-dir path/to/images --output-csv outputs/predictions.csv`
   `wildfire-cli --eval-dir data/test --metrics-json outputs/metrics.json`

5) Train
   `wildfire-train --config params.yaml`

Artifacts
- `artifacts/model_best.pth`
- `artifacts/model_latest.pth`
- `artifacts/metrics.json`
- `artifacts/reference_stats.json`

Configuration (env)
- `WILDFIRE_MODEL_ARCH=custom_cnn | resnet18`
- `WILDFIRE_MODEL_PATH_CUSTOM=wildfire_model.pth`
- `WILDFIRE_MODEL_PATH_RESNET18=artifacts/exp3/model_best.pth`
- `WILDFIRE_REFERENCE_STATS_PATH=artifacts/reference_stats.json`

Experiment Tracking (MLflow)
- Runs logged to `mlruns/` by default
- Launch UI: `mlflow ui`

Data Versioning (DVC)
- Track data: `dvc add data`
- Reproduce pipeline: `dvc repro -s train`

Monitoring (Drift)
- Reference stats saved during training
- Inference computes a lightweight drift score (non‑blocking)

Testing
- `pytest -q` for fast checks
- Includes API and training smoke tests
