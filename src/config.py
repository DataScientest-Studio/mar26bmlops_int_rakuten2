"""
Central config for Rakuten Color Extraction Project
Single source of truth for:
- Paths
- Model configs
- MinIO / object storage
- PostgreSQL / DB
- MLflow
- Training settings
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# ENV
# ──────────────────────────────────────────────────────────────
load_dotenv()

from dotenv import load_dotenv


# ──────────────────────────────────────────────────────────────
# ENV + Data
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
# MINIO / OBJECT STORAGE
# ──────────────────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")

# Prefer dedicated MinIO vars, but stay backward-compatible
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD", "password123")

# Backward-compatible aliases for existing code / compose
MINIO_ROOT_USER = MINIO_ACCESS_KEY
MINIO_ROOT_PASSWORD = MINIO_SECRET_KEY

MINIO_BUCKET_DATA = os.getenv("MINIO_BUCKET_DATA", "data")
MINIO_BUCKET_IMAGES = os.getenv("MINIO_BUCKET_IMAGES", "images")

# MINIO / OBJECT STORAGE  (merged from VR branch)
# ──────────────────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
 
MINIO_BUCKET_DATA = os.getenv("MINIO_BUCKET_DATA", "data")
MINIO_BUCKET_IMAGES = os.getenv("MINIO_BUCKET_IMAGES", "images")
 
MINIO_X_TRAIN_KEY = os.getenv("MINIO_X_TRAIN_KEY", "X_train.csv")
MINIO_Y_TRAIN_KEY = os.getenv("MINIO_Y_TRAIN_KEY", "y_train.csv")
MINIO_X_TEST_KEY = os.getenv("MINIO_X_TEST_KEY", "X_test.csv")
MINIO_Y_RANDOM_KEY = os.getenv("MINIO_Y_RANDOM_KEY", "y_random.csv")

# Image loading control
IMAGE_SOURCE = os.getenv("IMAGE_SOURCE", "minio").lower()   # "minio" or "local"
MINIO_IMAGE_PREFIX = os.getenv("MINIO_IMAGE_PREFIX", "")    # e.g. "train_images/"
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", str(DATA_DIR / "images")))

# Ensure endpoint has a usable HTTP URI when needed
if MINIO_ENDPOINT.startswith("http://") or MINIO_ENDPOINT.startswith("https://"):
    MINIO_HTTP_ENDPOINT = MINIO_ENDPOINT
else:
    MINIO_HTTP_ENDPOINT = f"http://{MINIO_ENDPOINT}"

MINIO_DATA_BASE_URI = f"s3://{MINIO_BUCKET_DATA}"

# ──────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────
DB_BACKEND = os.getenv("DB_BACKEND", "postgres").lower()

DATABASE_PATH = os.getenv("DATABASE_PATH", str(DB_DIR / "rakuten.db"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rakuten")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rakuten_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rakuten_pass")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)
 
# Image loading control: "minio" = stream from S3, "local" = read from disk
IMAGE_SOURCE = os.getenv("IMAGE_SOURCE", "local").lower()   # "minio" or "local"
DATA_SOURCE = os.getenv("DATA_SOURCE", "local").lower()     # "minio" or "local"
MINIO_IMAGE_PREFIX = os.getenv("MINIO_IMAGE_PREFIX", "")    # e.g. "train_images/"
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", str(DATA_DIR / "images")))
 
MINIO_HTTP_ENDPOINT = f"http://{MINIO_ENDPOINT}"
MINIO_DATA_BASE_URI = f"s3://{MINIO_BUCKET_DATA}"



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
    "image_dir": IMAGE_DIR,
    "image_source": IMAGE_SOURCE,
    "minio_bucket_images": MINIO_BUCKET_IMAGES,
    "minio_image_prefix": MINIO_IMAGE_PREFIX,
    "image_dir": DATA_DIR / "images",
}

# ──────────────────────────────────────────────────────────────
# ICE (Image-Caption Ensemble)
# ──────────────────────────────────────────────────────────────
ICE_CONFIG = {
    "text_model_id": os.getenv("TEXT_MODEL_ID", "cl-tohoku/bert-base-japanese-v3"),
    "vision_model_id": os.getenv("VISION_MODEL_ID", "openai/clip-vit-base-patch16"),

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

    "val_threshold": float(os.getenv("ICE_VAL_THRESHOLD", "0.5")),
    "train_threshold": float(os.getenv("ICE_TRAIN_THRESHOLD", "0.5")),

    # Thresholds
    "val_threshold": float(os.getenv("ICE_VAL_THRESHOLD", "0.5")),
    "train_threshold": float(os.getenv("ICE_TRAIN_THRESHOLD", "0.5")),

    # Data
    "max_len": int(os.getenv("ICE_MAX_LEN", "128")),
    "image_dir": IMAGE_DIR,
    "image_source": IMAGE_SOURCE,
    "minio_bucket_images": MINIO_BUCKET_IMAGES,
    "minio_image_prefix": MINIO_IMAGE_PREFIX,

    "db_train": os.getenv("DB_TRAIN_SPLIT", "train"),
    "predict_split": os.getenv("PREDICT_SPLIT", "val"),
    "val_ratio": float(os.getenv("VAL_RATIO", "0.1")),

    # Outputs
    "checkpoint_path": MODEL_DIR / "color_model_best.pth",
    "mlb_path": MODEL_DIR / "mlb.pkl",
    "predictions_path": MODEL_DIR / "y_pred_training.csv",
}

# ──────────────────────────────────────────────────────────────
# ENSEMBLE
# ENSEMBLE (optional future)
# ──────────────────────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    "text_weight": 0.65,
    "clip_weight": 0.35,
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

MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", MINIO_HTTP_ENDPOINT)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", MINIO_ACCESS_KEY)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", MINIO_SECRET_KEY)
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", MINIO_ROOT_USER)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", MINIO_ROOT_PASSWORD)

# ──────────────────────────────────────────────────────────────
# APP ENV
# ──────────────────────────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "development")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def export_params():


# ──────────────────────────────────────────────────────────────
# params.yaml create, Copy from ICE_CONFIG to params.yaml for DVC due DVC can Not read pyfiles
# ──────────────────────────────────────────────────────────────


import yaml  # pip install pyyaml if missing
 
def export_params():
    """Sync params.yaml for DVC - called automatically by pipeline."""
    params = {
        "ICE_CONFIG": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in ICE_CONFIG.items()
            if isinstance(v, (str, int, float, bool, Path))
        },
        "MINIO_CONFIG": {
            "endpoint": MINIO_ENDPOINT,
            "http_endpoint": MINIO_HTTP_ENDPOINT,
            "bucket_data": MINIO_BUCKET_DATA,
            "bucket_images": MINIO_BUCKET_IMAGES,
            "image_source": IMAGE_SOURCE,
            "image_prefix": MINIO_IMAGE_PREFIX,
            "x_train_key": MINIO_X_TRAIN_KEY,
            "y_train_key": MINIO_Y_TRAIN_KEY,
            "x_test_key": MINIO_X_TEST_KEY,
            "y_random_key": MINIO_Y_RANDOM_KEY,
        },
        "DATABASE_CONFIG": {
            "db_backend": DB_BACKEND,
            "database_url": DATABASE_URL,
            "postgres_host": POSTGRES_HOST,
            "postgres_port": POSTGRES_PORT,
            "postgres_db": POSTGRES_DB,
            "postgres_user": POSTGRES_USER,
        },
        "MLFLOW_CONFIG": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment": MLFLOW_EXPERIMENT,
            "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
            "champion_alias": MLFLOW_CHAMPION_ALIAS,
            "candidate_alias": MLFLOW_CANDIDATE_ALIAS,
            "s3_endpoint_url": MLFLOW_S3_ENDPOINT_URL,
        },
    }

            "s3_endpoint_url": MLFLOW_S3_ENDPOINT_URL,
        },
    }
    with open(PROJECT_ROOT / "params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)