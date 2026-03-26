"""
Zentrale Konfiguration fuer das Rakuten Color Extraction Projekt.
"""
from pathlib import Path

# ── Pfade ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
SUBMIT_DIR   = PROJECT_ROOT / "submissions"
DB_DIR       = PROJECT_ROOT / "db"

# Verzeichnisse sicherstellen
for d in [DATA_DIR, MODEL_DIR, SUBMIT_DIR, DB_DIR]:
    d.mkdir(exist_ok=True)

# ── Farb-Labels ────────────────────────────────────────────────
COLOR_LABELS = [
    'Black', 'White', 'Grey', 'Navy', 'Blue', 'Red', 'Pink',
    'Brown', 'Beige', 'Green', 'Khaki', 'Orange', 'Yellow',
    'Purple', 'Burgundy', 'Gold', 'Silver', 'Transparent', 'Multiple Colors'
]
NUM_LABELS = len(COLOR_LABELS)

# ── XLM-RoBERTa Defaults ──────────────────────────────────────
XLM_CONFIG = {
    "model_name":        "xlm-roberta-base",
    "lr":                2e-5,
    "epochs":            3,
    "batch_size":        32,
    "max_len":           256,
    "dropout":           0.3,
    "pos_weight_factor": 2.0,
    "weight_decay":      0.01,
    "warmup_ratio":      0.1,
    "grad_clip":         1.0,
}

# ── CLIP Defaults ──────────────────────────────────────────────
CLIP_CONFIG = {
    "model_name":  "ViT-B-32",
    "pretrained":  "laion2b_s34b_b79k",
    "batch_size":  64,
    "image_dir":   DATA_DIR / "images",
}

# ── ICE (Image-Caption Ensemble) – DualEncoder ────────────────
ICE_CONFIG = {
    "text_model_id":   "cl-tohoku/bert-base-japanese-v3",
    "vision_model_id": "openai/clip-vit-base-patch16",
    "batch_size":      128,
    "learning_rate":   3e-3,
    "encoder_lr":      2e-5,
    "max_epochs":      1,
    "unfreeze_layers": 2,
    "es_patience":     5,
    "val_threshold":   0.5,
    "train_threshold": 0.5,
    "max_len":         128,
    "image_dir":       DATA_DIR / "images",
    "db_train":        "train",
    # "val", "pseudo_test", oder "test" für Challenge
    "predict_split": "val",
    # Checkpoint-Dateien
    "checkpoint_path": MODEL_DIR / "color_model_best.pth",
    "mlb_path":        MODEL_DIR / "mlb.pkl",
}

# ── Ensemble ───────────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    "text_weight": 0.65,
    "clip_weight": 0.35,
}

# ── MLflow ─────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "rakuten_color_extraction"
MLFLOW_TRACKING_URI = "http://localhost:5000"
