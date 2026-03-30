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
    ├── reports            <- Generated analyses and reports
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

## Phase 2 Preparation
- MLflow is set up for experiment tracking

- Environment variables are defined in .env

- API entry point: src/api/main.py

- Raw and processed data are excluded from version control

- Development is done via feature branches

## Local setup
```bash
cp .env
pip install -r requirements.txt
uvicorn src.api.main:app --reload
pytest
```

## Start MLflow
```bash
mlflow server --host 0.0.0.0 --port 5000
```
## Phase 2 Overview
- MLflow experiment tracking is integrated in the training service.
- Model versioning is handled with MLflow Model Registry.
- The best model is selected by comparing the new retrained model against the previous champion on validation micro-F1.
- The application is decomposed into Docker microservices: mlflow, training, api.

## How to run (Docker)
1. docker compose up -d mlflow
2. docker compose up -d api
3. docker compose run --rm training python -m src.models.train_model_ice_mk

## Services
- MLflow UI: http://localhost:5000
- API: http://localhost:8000

## Model Registry
- Registered model name: rakuten-ice-dual-encoder
   - Candidate alias: candidate
   - Champion alias: champion
