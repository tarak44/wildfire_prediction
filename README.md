# Wildfire Risk Intelligence System

Multimodal wildfire risk prediction platform built with PyTorch, FastAPI, MLflow, Docker, and Streamlit.

This project treats wildfire detection as a risk intelligence problem, not a toy image classification task. It combines:

- visual evidence from wildfire imagery
- structured environmental features such as temperature, humidity, wind, drought, and fuel dryness
- optional temporal weather sequences
- explainability with Grad-CAM
- production-style serving, tracking, benchmarking, and deployment paths

## Why This Project Is Different

Most wildfire ML demos stop at binary image classification. This repository is designed like a real applied ML system:

- Trained multimodal fusion: image encoder + tabular encoder + fusion head, instead of heuristic score blending
- Temporal context support: LSTM or Temporal MLP over weather history
- Production API: single inference, batch inference, health checks, model metadata, structured errors
- MLOps discipline: MLflow tracking, model registration hooks, Docker packaging, reproducible configs
- Interpretability: Grad-CAM overlays plus human-readable risk-factor summaries

## Problem Framing

A useful wildfire system should answer more than:

> "Does this image look like fire?"

It should answer:

> "Given the image, current weather, and recent environmental conditions, what is the wildfire risk, how confident is the model, and which signals drove the decision?"

That is the framing of this project.

## System Architecture

```text
                                 +-----------------------------+
                                 | External Data Sources       |
                                 | weather / drought / GIS /   |
                                 | satellite / fuel moisture   |
                                 +-------------+---------------+
                                               |
                                               v
+-------------------+              +---------------------------+
| Image Sources     | -----------> | Ingestion Layer           |
| camera / UAV /    |              | manifests / APIs / files  |
| analyst uploads   |              +-------------+-------------+
+-------------------+                            |
                                                 v
                                 +-----------------------------+
                                 | Preprocessing               |
                                 | image transforms            |
                                 | tabular normalization       |
                                 | temporal sequence parsing   |
                                 +-------------+---------------+
                                               |
                                               v
                                 +-----------------------------+
                                 | Training Layer              |
                                 | CNN / EfficientNet          |
                                 | Multimodal fusion           |
                                 | Temporal encoder            |
                                 | class weighting             |
                                 +-------------+---------------+
                                               |
                                               v
                                 +-----------------------------+
                                 | Experiment Tracking         |
                                 | MLflow + registry           |
                                 | metrics + artifacts         |
                                 +-------------+---------------+
                                               |
                 +-----------------------------+-----------------------------+
                 |                                                           |
                 v                                                           v
    +-----------------------------+                           +-----------------------------+
    | FastAPI Inference Service   |                           | Streamlit Frontend          |
    | /predict                    |                           | upload + risk + Grad-CAM    |
    | /predict-batch              |                           | analyst demo surface        |
    | /health /model-info         |                           +-----------------------------+
    +-----------------------------+
```

## Models

### 1. Baseline Vision Models

- `custom_cnn`
- `resnet18`
- `efficientnet_b0`

These are useful for benchmarking and for showing progression from baseline to transfer learning.

### 2. Trained Multimodal Model

Implemented in [`multimodal.py`](/c:/Users/thara/Wildfire_Prediction/src/wildfire_mlops/modeling/multimodal.py).

Architecture:

- Image encoder: EfficientNet-B0 backbone projected to a compact image embedding
- Tabular encoder: MLP over environmental features
- Temporal encoder: optional `LSTM` or `Temporal MLP`
- Fusion head: concatenation of image embedding + context embedding, followed by dense layers
- Auxiliary context head: predicts wildfire risk from structured context alone to stabilize multimodal learning

### 3. Loss Function

Main loss:

```text
CrossEntropy(fusion_logits, y)
```

Auxiliary context loss:

```text
CrossEntropy(context_logits, y)
```

Total loss:

```text
total_loss = main_loss + alpha * context_loss
```

This helps the structured branch learn useful context representations instead of becoming a passive side input.

## Temporal Modeling

Temporal context is supported for use cases like rolling weather windows, hourly drought signals, or recent wind changes.

Two options are implemented:

- `lstm`: better when order matters and you have a real hourly or daily sequence
- `temporal_mlp`: simpler and faster when you want a strong tabular baseline over fixed windows

### Temporal Input Format

Each training row can include a `temporal_sequence` column containing JSON such as:

```json
[
  {"temperature_c": 31.2, "humidity_pct": 26.0, "wind_speed_kph": 18.0, "drought_index": 380.0, "vegetation_dryness": 0.68},
  {"temperature_c": 33.1, "humidity_pct": 23.0, "wind_speed_kph": 22.0, "drought_index": 390.0, "vegetation_dryness": 0.70},
  {"temperature_c": 35.0, "humidity_pct": 21.0, "wind_speed_kph": 25.0, "drought_index": 405.0, "vegetation_dryness": 0.73}
]
```

The loader parses this into `[seq_len, feature_dim]` tensors and pads batches correctly for the temporal encoder.

## Data Schema For Multimodal Training

Example manifest row:

```csv
image_path,label,temperature_c,humidity_pct,wind_speed_kph,drought_index,vegetation_dryness,days_since_rain,temporal_sequence
data/images/sample_001.jpg,wildfire,35.0,21.0,24.0,410.0,0.78,11,"[{""temperature_c"":32.0,""humidity_pct"":28.0,""wind_speed_kph"":15.0,""drought_index"":390.0,""vegetation_dryness"":0.72}]"
```

Relevant config:

- [`configs/experiments/multimodal.yaml`](/c:/Users/thara/Wildfire_Prediction/configs/experiments/multimodal.yaml)

Generate the manifests directly from the image folders:

```bash
wildfire-generate-multimodal-data --data-root data --output-dir data/manifests --temporal-steps 4
```

## Training

### Image-Only Training

```bash
wildfire-train --config configs/experiments/exp3.yaml
```

### Multimodal Training

```bash
wildfire-train-multimodal --config configs/experiments/multimodal.yaml
```

### What The Multimodal Trainer Handles

- image + tabular + temporal loading
- class-weighted loss for imbalance
- macro metrics
- ROC-AUC
- checkpoint metadata for later inference
- MLflow logging and optional registry registration

## Evaluation And Benchmarking

### Metrics Tracked

- accuracy
- balanced accuracy
- precision
- recall
- F1
- macro precision
- macro recall
- macro F1
- ROC-AUC
- confusion matrix

### Current Benchmark Snapshot

Measured from existing repo artifacts:

| Model | Source | Accuracy | Notes |
|---|---|---:|---|
| Custom CNN | [`artifacts/metrics.json`](/c:/Users/thara/Wildfire_Prediction/artifacts/metrics.json) | 93.10% test | strong baseline with class weighting |
| ResNet18 transfer learning | [`artifacts/exp3/metrics.json`](/c:/Users/thara/Wildfire_Prediction/artifacts/exp3/metrics.json) | 95.75% validation | best currently tracked vision model |
| EfficientNet-B0 | [`artifacts/exp4/metrics.json`](/c:/Users/thara/Wildfire_Prediction/artifacts/exp4/metrics.json) | 98.03% test | strongest image-only model in current benchmark |
| Multimodal EfficientNet + MLP | [`artifacts/multimodal/multimodal_metrics.json`](/c:/Users/thara/Wildfire_Prediction/artifacts/multimodal/multimodal_metrics.json) | 99.78% test | best overall model with synthetic environmental context |

### Generate A Comparison Table

```bash
wildfire-benchmark \
  --entry custom_cnn=artifacts/metrics.json \
  --entry resnet18=artifacts/exp3/metrics.json \
  --entry multimodal=artifacts/multimodal/multimodal_metrics.json \
  --output outputs/benchmark.md
```

The benchmark utility renders a markdown table suitable for a README, report, or PR.

The current generated comparison table is in [`outputs/benchmark.md`](/c:/Users/thara/Wildfire_Prediction/outputs/benchmark.md).

## FastAPI Backend

Implemented in [`api/main.py`](/c:/Users/thara/Wildfire_Prediction/api/main.py).

### Endpoints

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`

### `GET /model-info`

Returns:

- active model architecture
- model path
- supported context modes
- class names
- MLflow tracking URI
- model version

### `POST /predict`

Form inputs:

- image file
- optional weather / environmental fields
- optional `temporal_context_json`
- optional `include_explainability`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample.jpg" \
  -F "temperature_c=36" \
  -F "humidity_pct=18" \
  -F "wind_speed_kph=25" \
  -F "drought_index=420" \
  -F "vegetation_dryness=0.81" \
  -F "days_since_rain=12" \
  -F "include_explainability=true"
```

### `POST /predict-batch`

JSON input for batch scoring:

```json
{
  "items": [
    {
      "image_base64": "<base64-encoded-image>",
      "environmental_context": {
        "temperature_c": 36.0,
        "humidity_pct": 18.0,
        "wind_speed_kph": 25.0,
        "drought_index": 420.0,
        "vegetation_dryness": 0.81,
        "days_since_rain": 12.0
      },
      "include_explainability": false
    }
  ]
}
```

### Error Handling

The API returns structured 4xx and 5xx responses for:

- invalid images
- malformed temporal context JSON
- invalid base64 payloads
- missing multimodal context for trained multimodal checkpoints

## Explainability

The inference layer produces:

- Grad-CAM overlay image
- prediction confidence
- contextual factor ranking
- short natural-language explanation summary

This is useful for:

- model debugging
- analyst review
- stakeholder demos
- detecting spurious visual shortcuts

## Streamlit Frontend

Run locally:

```bash
streamlit run apps/streamlit_app.py
```

The app supports:

- image upload
- architecture selection
- structured weather inputs
- confidence and risk display
- Grad-CAM visualization

Generate evaluation plots:

```bash
wildfire-visualize \
  --manifest data/manifests/test_multimodal.csv \
  --model-path artifacts/multimodal/multimodal_model_best.pth \
  --model-arch temporal_multimodal_efficientnet_b0 \
  --output-dir outputs/figures/multimodal
```

## MLflow Production Setup

The repo now defaults to SQLite-backed MLflow rather than the deprecated local file store.

### Option A: SQLite

Run the tracking server locally:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Point training to it:

```bash
set WILDFIRE_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
wildfire-train --config configs/experiments/exp3.yaml
```

### Option B: PostgreSQL

Start Postgres:

```bash
docker run --name wildfire-mlflow-postgres ^
  -e POSTGRES_USER=mlflow ^
  -e POSTGRES_PASSWORD=mlflow ^
  -e POSTGRES_DB=mlflow ^
  -p 5432:5432 ^
  -d postgres:16
```

Start MLflow against Postgres:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql+psycopg2://mlflow:mlflow@127.0.0.1:5432/mlflow \
  --default-artifact-root ./mlartifacts
```

### Tracking And Registry Best Practices

- one experiment per modeling family: `wildfire-baseline`, `wildfire-transfer-learning`, `wildfire-multimodal`
- log dataset version, feature schema, and model architecture as params
- store macro F1 and ROC-AUC as primary selection metrics
- register only validated checkpoints
- use model aliases like `Champion` and `Shadow` once registry workflows mature

## Deployment

### Local Docker

API:

```bash
docker build -f docker/Dockerfile.api -t wildfire-api .
docker run --rm -p 8000:8000 wildfire-api
```

Frontend:

```bash
docker build -f docker/Dockerfile.streamlit -t wildfire-streamlit .
docker run --rm -p 8501:8501 wildfire-streamlit
```

### Public Deployment Layout

- Render API blueprint: [`render.yaml`](/c:/Users/thara/Wildfire_Prediction/render.yaml)
- API container: [`docker/Dockerfile.api`](/c:/Users/thara/Wildfire_Prediction/docker/Dockerfile.api)
- Hugging Face Space bundle: [`deploy/huggingface-space`](/c:/Users/thara/Wildfire_Prediction/deploy/huggingface-space)

Suggested production split:

- Render hosts the FastAPI inference service
- Hugging Face Docker Space hosts the Streamlit frontend and calls the Render API with `WILDFIRE_API_URL`

Deployment note:

- the API container uses the tracked deployment bundle in `deploy/models/` so public builds do not depend on ignored local training artifacts

### Deployment Links

Add your deployed URLs here once live:

- API: `https://<your-render-or-railway-api>.onrender.com`
- Frontend: `https://huggingface.co/spaces/<username>/wildfire-risk-intelligence`
- MLflow UI: `https://<internal-or-private-mlflow-host>`

Example badge once live:

```md
[![Live Demo](https://img.shields.io/badge/demo-live-red)](https://huggingface.co/spaces/<username>/wildfire-risk-intelligence)
```

## Scaling Strategy

### Near-Term

- async FastAPI handlers
- `asyncio.to_thread` for CPU-bound batch requests
- horizontal API replicas behind a load balancer

### Next Step

Introduce a queue-backed inference path:

- FastAPI receives request
- request metadata goes to Redis or another queue
- worker pool runs model inference
- results stored in cache or database

Practical choices:

- Celery + Redis
- RQ + Redis
- managed cloud queue if deploying to a cloud platform

This is the right direction for:

- bursty batch uploads
- large wildfire imagery feeds
- longer multimodal or temporal inference pipelines

## Local Development

Install:

```bash
pip install -e .
```

Run tests:

```bash
python -m pytest -q
python -m ruff check api apps src tests
```

## Project Structure

```text
api/                               FastAPI inference service
apps/                              Streamlit demo
configs/                           training configs
docker/                            Dockerfiles
src/wildfire_mlops/modeling/       CNN, EfficientNet, multimodal models
src/wildfire_mlops/data/           image and multimodal datasets
src/wildfire_mlops/training/       training, metrics, benchmark utilities
tests/                             smoke and multimodal tests
```

## License

MIT
