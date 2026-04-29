# Rakuten Product Color Classification - MLOps Project

This project is part of an MLOps course project.  
The goal is to build a reproducible machine learning pipeline for **multi-label product color classification** based on Rakuten product data.

## Project Scope
Phase 1 includes:
- project setup and environment configuration
- data loading and preprocessing
- baseline model training
- basic inference API

## Project Organization

    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── external       <- Data from third party sources
    │   ├── interim        <- Intermediate transformed data
    │   ├── processed      <- Final processed datasets used for modeling
    │   └── raw            <- Original raw input data
    │
    ├── logs               <- Logs from training and prediction
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks for exploration
    │
    ├── references         <- Data dictionaries and project materials
    │
    <!-- ├── reports            <- Generated analyses and reports -->
    │   └── figures        <- Generated graphics and figures
    │
    ├── requirements.txt   <- Python dependencies
    │
    ├── src                <- Source code
    │   ├── __init__.py
    │   ├── data           <- Data loading and preprocessing scripts
    │   │   └── load_data.py
    │   ├── features       <- Feature engineering scripts
    │   ├── models         <- Training and prediction scripts
    │   ├── visualization  <- Visualization scripts
    │   └── config         <- Configuration files

## PHASE: 1 Setup

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

```
## Install dependencies:
```bash
python -m pip install -r requirements.txt
```
## Run preprocessing
```bash
python src/data/load_data.py
```
## Current preprocessing steps

- load X_train.csv, y_train.csv, and X_test.csv

- drop auto-generated index columns if present

- fill missing values in item_name and item_caption

- combine text fields into combined_text

- parse color_tags into Python lists

- save processed files into data/processed/

## Current dataset summary

- training samples: 212,120

- test samples: 37,347

- number of unique color labels: 19

- task type: multi-label classification

## Next steps

- build a baseline model using TF-IDF + Logistic Regression

- implement training.py and predict.py

- create a basic FastAPI inference service

## # Phase 2 – MLflow Tracking, Registry & Storage

This project uses a production-style MLflow setup with PostgreSQL and MinIO.

## Architecture

- **MLflow Tracking Server**: experiment logging and model registry
- **PostgreSQL**: backend metadata store for MLflow
- **MinIO (S3-compatible)**: artifact storage for models, datasets, and outputs
- **Training Service**: trains and evaluates models
- **API Service**: serves predictions using promoted models

## Environment Configuration

Main variables:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTERED_MODEL_NAME=rakuten-ice-dual-encoder

MLFLOW_BACKEND_STORE_URI=postgresql://postgres:postgres@postgres:5432/mlflow_meta

MLFLOW_S3_ENDPOINT_URL=http://minio:9000

AWS_ACCESS_KEY_ID=admin
AWS_SECRET_ACCESS_KEY=...
```

## Experiment Tracking

Each training run logs:

hyperparameters
batch size
learning rates
max epochs
thresholds
validation metrics
micro-F1 / macro-F1
trained model artifacts
label binarizer artifacts

## Model Registry Workflow

The registered model:
- rakuten-ice-dual-encoder

## Aliases used:

- candidate
- champion

# Workflow:

1. Train new model
2. Log run to MLflow
3. Register model version
4. Set alias candidate
5. Compare with current champion
6. Promote better model automatically

## Scripts

# Training
```bash
python -m src.models.train_model_final
```
# Prediction
```bash
python -m src.models.predict_model_final
```

# Compare & Promote
```bash
python -m src.models.compare_and_promote
```
# Full Pipeline
```bash
python -m src.pipeline --mode full
```

# Docker Usage
```bash
docker compose up -d
docker compose run --rm training python -m src.pipeline --mode train
```
## Services
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001
- API: http://localhost:8000

## Storage Strategy
- Structured metadata → PostgreSQL
- Models & artifacts → MinIO
- Registry versions → MLflow Model Registry

## # Phase 3 – Monitoring Stack (Prometheus, Grafana, Evidently)

This phase adds full observability over both the running API and the training pipeline.

## Architecture

- **Prometheus**: scrapes metrics from the API and Pushgateway every 15 seconds
- **Pushgateway**: receives training metrics pushed at the end of each Airflow pipeline run
- **Grafana**: visualizes all metrics via auto-provisioned dashboards (no manual setup required)
- **Evidently**: generates interactive HTML data-drift reports comparing reference vs. current data

## Two Metric Flows

**API metrics** — emitted continuously by the running API:

| Metric | Type | Labels |
|---|---|---|
| `rakuten_color_predictions_total` | Counter | color |
| `rakuten_requests_total` | Counter | method, endpoint, status_code |
| `rakuten_request_duration_seconds` | Histogram | method, endpoint |

**Training metrics** — pushed once per Airflow run via Pushgateway:

| Metric | Type | Labels |
|---|---|---|
| `rakuten_training_run_f1` | Gauge | model_version, run_id |
| `rakuten_training_duration_seconds` | Gauge | model_version |
| `rakuten_champion_f1` | Gauge | — |
| `rakuten_champion_version` | Gauge | — |

## Grafana Dashboards

Dashboards are provisioned automatically from `grafana/provisioning/`:

- **API Monitoring** — request counts, durations, color prediction rates
- **Training Monitoring** — per-run F1, training duration, champion F1 and version

## Data Drift with Evidently

Evidently compares a reference dataset against a current dataset and runs statistical tests per feature. It outputs a drift score and verdict per feature, and generates an interactive HTML report. In a production setup, drift detection would run periodically and trigger retraining via the Airflow pipeline if drift is detected.

Generate a drift report:
```bash
python -m src.monitoring.drift
```

## Services
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Pushgateway: http://localhost:9091
