"""
main pipline

Use:
        python -m src.pipeline --mode ingest      # fill only DB

"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    COLOR_LABELS, DATA_DIR
)

from src.db import init_db, ingest_products, get_db_summary


def run_pipeline(mode="full"):
    """
    exectes the pipline

    Args:
        mode: 'full', 'ingest', 'train', 'predict'
    """
    print("=" * 60)
    print("RAKUTEN COLOR EXTRACTION PIPELINE")
    print(f"  Mode: {mode}")
    print("=" * 60)

    # ── 1. Data loading ──────────────────────────────────────
    print("\n[1/X] Data loading...")
    df_x    = pd.read_csv(DATA_DIR / "raw" / "X_train.csv")
    df_y    = pd.read_csv(DATA_DIR / "raw" / "y_train.csv")
    df_test = pd.read_csv(DATA_DIR / "raw" / "X_test.csv")

    print(f"  Train: {len(df_x)}, Test: {len(df_test)}")

    # ── 2. Splits ───────────────────────────────────────────
    print("\n[2/X] Train/Val/Pseudo-Test Split...")
    train_x, temp_x, train_y, temp_y = train_test_split(
        df_x, df_y, test_size=0.2, random_state=42
    )
    val_x, pseudo_x, val_y, pseudo_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=42
    )
    print(f"  Train={len(train_x)}, Val={len(val_x)}, Pseudo={len(pseudo_x)}")

    # ── 3. DB filling ─────────────────────────────────────
    if mode == "full":
        print("\n[3/X] Datenbank befuellen...")
        init_db()
        ingest_products(train_x, train_y,   split="train")
        ingest_products(val_x,   val_y,     split="val")
        ingest_products(pseudo_x, pseudo_y, split="pseudo_test")
        ingest_products(df_test, df_y=None,  split="test")

        summary = get_db_summary()
        print(f"  DB: {summary['products_by_split']}")

    if mode == "ingest":
        init_db()
        ingest_products(df_x, df_y, split="train")
        print("\nReady (Ingest x_train only).")
        return


if __name__ == "__main__":
    run_pipeline(mode="full")
