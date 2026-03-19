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