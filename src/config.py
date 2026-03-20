"""
central config for the whole Racuten color detect project
"""

from pathlib import Path

# ── paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
SUBMIT_DIR   = PROJECT_ROOT / "submissions"
DB_DIR       = PROJECT_ROOT / "db"

# Secure Dir exists
for d in [DATA_DIR, MODEL_DIR, SUBMIT_DIR, DB_DIR]:
    d.mkdir(exist_ok=True)


# ── Color-Labels ────────────────────────────────────────────────

COLOR_LABELS = [
    'Black', 'White', 'Grey', 'Navy', 'Blue', 'Red', 'Pink',
    'Brown', 'Beige', 'Green', 'Khaki', 'Orange', 'Yellow',
    'Purple', 'Burgundy', 'Gold', 'Silver', 'Transparent', 'Multiple Colors'
]

NUM_LABELS = len(COLOR_LABELS)


XLM_CONFIG = {
    "model_name":           "xlm-roberta-base",
    "lr":                   2e-5,
    "epochs":               3,
    "batch_size":           32,
    "max_len":              256,
    "dropout":              0.3,
    "pos_weight_factor":    2.0,
    "weight_decay":         0.01,
    "warmup_ratio":         0.1,
    "grad_clip":            1.0,
}
