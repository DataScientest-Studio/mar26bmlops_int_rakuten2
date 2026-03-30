"""
Central config for Rakuten Color Extraction Project
Single source of truth for:
- Paths
- Model configs
- MLflow
- Training settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# ENV
# ──────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SUBMIT_DIR = PROJECT_ROOT / "submissions"
DB_DIR = PROJECT_ROOT / "db"

# Ensure folders exist
for d in [DATA_DIR, MODEL_DIR, SUBMIT_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# COLOR LABELS
# ──────────────────────────────────────────────────────────────
COLOR_LABELS = [
    "Black", "White", "Grey", "Navy", "Blue", "Red", "Pink",
    "Brown", "Beige", "Green", "Khaki", "Orange", "Yellow",
    "Purple", "Burgundy", "Gold", "Silver", "Transparent", "Multiple Colors"
]

NUM_LABELS = len(COLOR_LABELS)

# ──────────────────────────────────────────────────────────────
# TEXT MODEL (XLM)
# ──────────────────────────────────────────────────────────────
XLM_CONFIG = {
    "model_name": "xlm-roberta-base",
    "lr": 2e-5,
    "epochs": 5,
    "batch_size": 32,
    "max_len": 256,
    "dropout": 0.3,
    "pos_weight_factor": 2.0,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
}

# ──────────────────────────────────────────────────────────────
# CLIP MODEL
# ──────────────────────────────────────────────────────────────
CLIP_CONFIG = {
    "model_name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "batch_size": 64,
    "image_dir": DATA_DIR / "images",
}

# ──────────────────────────────────────────────────────────────
# ICE (Image-Caption Ensemble)
# ──────────────────────────────────────────────────────────────
ICE_CONFIG = {
    # Models
    "text_model_id": os.getenv("TEXT_MODEL_ID", "cl-tohoku/bert-base-japanese-v3"),
    "vision_model_id": os.getenv("VISION_MODEL_ID", "openai/clip-vit-base-patch16"),

    # Training
    "batch_size": int(os.getenv("ICE_BATCH_SIZE", "128")),
    "learning_rate": float(os.getenv("ICE_LR", "3e-3")),
    "encoder_lr": float(os.getenv("ICE_ENCODER_LR", "2e-5")),
    "max_epochs": int(os.getenv("ICE_MAX_EPOCHS", "15")),
    "unfreeze_layers": int(os.getenv("ICE_UNFREEZE_LAYERS", "2")),
    "es_patience": int(os.getenv("ICE_ES_PATIENCE", "3")),

    # Thresholds
    "val_threshold": float(os.getenv("ICE_VAL_THRESHOLD", "0.5")),
    "train_threshold": float(os.getenv("ICE_TRAIN_THRESHOLD", "0.5")),

    # Data
    "max_len": int(os.getenv("ICE_MAX_LEN", "128")),
    "image_dir": Path(os.getenv("IMAGE_DIR", str(DATA_DIR / "images"))),
    "db_train": os.getenv("DB_TRAIN_SPLIT", "train"),
    "predict_split": os.getenv("PREDICT_SPLIT", "val"),
    "val_ratio": float(os.getenv("VAL_RATIO", "0.1")),

    # Outputs
    "checkpoint_path": MODEL_DIR / "color_model_best.pth",
    "mlb_path": MODEL_DIR / "mlb.pkl",
    "predictions_path": MODEL_DIR / "y_pred_training.csv",
}

# ──────────────────────────────────────────────────────────────
# ENSEMBLE (optional future)
# ──────────────────────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    "text_weight": 0.65,
    "clip_weight": 0.35,
}

# ──────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DB_DIR / "rakuten.db"))

# ──────────────────────────────────────────────────────────────
# MLFLOW
# ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "rakuten_color_extraction"
)

# Registry
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "rakuten-ice-dual-encoder"
)

MLFLOW_CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
MLFLOW_CANDIDATE_ALIAS = os.getenv("MLFLOW_CANDIDATE_ALIAS", "candidate")

# ──────────────────────────────────────────────────────────────
# APP ENV
# ──────────────────────────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "development")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")