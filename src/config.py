"""
Central config for Rakuten Color Extraction Project
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load project-local .env explicitly
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SUBMIT_DIR = PROJECT_ROOT / "submissions"
DB_DIR = PROJECT_ROOT / "db"

for d in [DATA_DIR, MODEL_DIR, SUBMIT_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# MINIO / OBJECT STORAGE
# ──────────────────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "pwd_123_SIMV")

# backward compatibility
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", MINIO_ACCESS_KEY)
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", MINIO_SECRET_KEY)

MINIO_BUCKET_DATA = os.getenv("MINIO_BUCKET_DATA", "data")
MINIO_BUCKET_IMAGES = os.getenv("MINIO_BUCKET_IMAGES", "images")

MINIO_X_TRAIN_KEY = os.getenv("MINIO_X_TRAIN_KEY", "X_train.csv")
MINIO_Y_TRAIN_KEY = os.getenv("MINIO_Y_TRAIN_KEY", "y_train.csv")
MINIO_X_TEST_KEY = os.getenv("MINIO_X_TEST_KEY", "X_test.csv")
MINIO_Y_RANDOM_KEY = os.getenv("MINIO_Y_RANDOM_KEY", "y_random.csv")

IMAGE_SOURCE = os.getenv("IMAGE_SOURCE", "minio").lower()
DATA_SOURCE = os.getenv("DATA_SOURCE", "minio").lower()

MINIO_IMAGE_PREFIX = os.getenv("MINIO_IMAGE_PREFIX", "")
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", str(DATA_DIR / "images")))

MINIO_HTTP_ENDPOINT = (
    MINIO_ENDPOINT
    if str(MINIO_ENDPOINT).startswith("http://") or str(MINIO_ENDPOINT).startswith("https://")
    else f"http://{MINIO_ENDPOINT}"
)

# ──────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────
DB_BACKEND = os.getenv("DB_BACKEND", "postgres")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rakuten")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# ──────────────────────────────────────────────────────────────
# COLOR LABELS
# ──────────────────────────────────────────────────────────────
COLOR_LABELS = [
    "Black",
    "White",
    "Grey",
    "Navy",
    "Blue",
    "Red",
    "Pink",
    "Brown",
    "Beige",
    "Green",
    "Khaki",
    "Orange",
    "Yellow",
    "Purple",
    "Burgundy",
    "Gold",
    "Silver",
    "Transparent",
    "Multiple Colors",
]

NUM_LABELS = len(COLOR_LABELS)

# ──────────────────────────────────────────────────────────────
# ICE MODEL CONFIG
# ──────────────────────────────────────────────────────────────
ICE_CONFIG = {
    "text_model_id": os.getenv("TEXT_MODEL_ID", "cl-tohoku/bert-base-japanese-v3"),
    "vision_model_id": os.getenv("VISION_MODEL_ID", "openai/clip-vit-base-patch16"),
    "batch_size": int(os.getenv("ICE_BATCH_SIZE", "256")),
    "learning_rate": float(os.getenv("ICE_LR", "0.001")),
    "encoder_lr": float(os.getenv("ICE_ENCODER_LR", "2e-5")),
    "max_epochs": int(os.getenv("ICE_MAX_EPOCHS", "1")),
    "unfreeze_layers": int(os.getenv("ICE_UNFREEZE_LAYERS", "2")),
    "es_patience": int(os.getenv("ICE_ES_PATIENCE", "3")),
    "val_threshold": float(os.getenv("ICE_VAL_THRESHOLD", "0.5")),
    "train_threshold": float(os.getenv("ICE_TRAIN_THRESHOLD", "0.5")),
    "max_len": int(os.getenv("ICE_MAX_LEN", "128")),
    "image_dir": IMAGE_DIR,
    "image_source": IMAGE_SOURCE,
    "minio_bucket_images": MINIO_BUCKET_IMAGES,
    "minio_image_prefix": MINIO_IMAGE_PREFIX,
    "db_train": os.getenv("DB_TRAIN_SPLIT", "train"),
    "predict_split": os.getenv("PREDICT_SPLIT", "val"),
    "val_ratio": float(os.getenv("VAL_RATIO", "0.1")),
    "checkpoint_path": MODEL_DIR / "color_model_best.pth",
    "mlb_path": MODEL_DIR / "mlb.pkl",
}

# ──────────────────────────────────────────────────────────────
# MLFLOW
# ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "rakuten_color_extraction",
)

MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "rakuten-ice-dual-encoder",
)

MLFLOW_CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
MLFLOW_CANDIDATE_ALIAS = os.getenv("MLFLOW_CANDIDATE_ALIAS", "candidate")

MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", MINIO_HTTP_ENDPOINT)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", MINIO_ACCESS_KEY)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", MINIO_SECRET_KEY)

# ──────────────────────────────────────────────────────────────
# EXPORT PARAMS FOR DVC
# ──────────────────────────────────────────────────────────────
def export_params():
    params = {
        "ICE_CONFIG": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in ICE_CONFIG.items()
        }
    }

    with open(PROJECT_ROOT / "params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, allow_unicode=True)