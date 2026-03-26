"""
Haupt-Pipeline: einmal aufrufen = alles laeuft durch.

Nutzung:
    python -m src.pipeline                    # Full Pipeline (Mock-Daten)
    python -m src.pipeline --real             # Mit echten Rakuten-Daten
    python -m src.pipeline --mode ingest      # Nur DB befuellen
    python -m src.pipeline --mode train       # Nur Training
    python -m src.pipeline --mode predict     # Nur Prediction
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    COLOR_LABELS, DATA_DIR, XLM_CONFIG, ENSEMBLE_CONFIG, MLFLOW_EXPERIMENT
)
from src.db import init_db, ingest_products, get_db_summary, save_predictions
from src.models.train_model import (
    train_xlm, evaluate, get_text_probs, optimize_thresholds,
    RakutenTextDataset, TextColorClassifier
)
from src.models.clip_model import clip_zero_shot_scores
from src.models.predict_model import predict_ensemble, create_submission


def run_pipeline(mode="full", real=False, config_overrides=None):
    """
    Fuehrt die gesamte Pipeline aus.

    Args:
        mode: 'full', 'ingest', 'train', 'predict'
        real: True = echte Rakuten-Daten, False = Mock-Daten
        config_overrides: Dict mit Config-Ueberschreibungen
    """
    print("=" * 60)
    print("RAKUTEN COLOR EXTRACTION PIPELINE")
    print(f"  Mode: {mode} | Daten: {'real' if real else 'mock'}")
    print("=" * 60)

    # ── 1. Daten laden ──────────────────────────────────────
    print("\n[1/8] Daten laden...")

    df_x    = pd.read_csv(DATA_DIR / "raw" / "X_train.csv")
    df_y    = pd.read_csv(DATA_DIR / "raw" / "y_train.csv")
    df_test = pd.read_csv(DATA_DIR / "raw" / "X_test.csv")

    print(f"  Train: {len(df_x)}, Test: {len(df_test)}")

    # ── 2. Splits ───────────────────────────────────────────
    print("\n[2/8] Train/Val/Pseudo-Test Split...")
    train_x, temp_x, train_y, temp_y = train_test_split(
        df_x, df_y, test_size=0.2, random_state=42
    )
    val_x, pseudo_x, val_y, pseudo_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=42
    )
    print(f"  Train={len(train_x)}, Val={len(val_x)}, Pseudo={len(pseudo_x)}")

    # ── 3. DB befuellen ─────────────────────────────────────
    if mode in ("full", "ingest"):
        print("\n[3/8] Datenbank befuellen...")
        init_db()
        ingest_products(train_x, train_y,   split="train")
        ingest_products(val_x,   val_y,     split="val")
        ingest_products(pseudo_x, pseudo_y, split="pseudo_test")
        ingest_products(df_test, df_y=None,  split="test")

        summary = get_db_summary()
        print(f"  DB: {summary['products_by_split']}")

    if mode == "ingest":
        print("\nFertig (nur Ingest).")
        return

    # ── 4. Text-Modell trainieren ────────────────────────────
    if mode in ("full", "train"):
        print("\n[4/8] XLM-RoBERTa Training...")
        config = {**XLM_CONFIG, **(config_overrides or {})}
        text_model, text_thresholds, text_run_id = train_xlm(
            train_x, train_y, val_x, val_y, config
        )

    # ── 5. CLIP Zero-Shot Scores ─────────────────────────────
    if mode in ("full", "train"):
        print("\n[5/8] CLIP Zero-Shot Scoring...")
        clip_val_probs = clip_zero_shot_scores(val_x)

    if mode in ("train",):
        print("\nFertig (nur Training).")
        return text_model, text_thresholds, text_run_id

    # ── 6. Ensemble: Late Fusion ─────────────────────────────
    if mode == "full":
        print("\n[6/8] Ensemble optimieren...")
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader

        tokenizer = AutoTokenizer.from_pretrained(config.get('model_name', 'xlm-roberta-base'))
        val_ds = RakutenTextDataset(val_x, val_y, tokenizer, config.get('max_len', 256))
        val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
        _, text_val_probs = evaluate(text_model, val_dl)

        alpha = ENSEMBLE_CONFIG['text_weight']
        ensemble_probs = alpha * text_val_probs + (1 - alpha) * clip_val_probs
        ens_thresholds = optimize_thresholds(ensemble_probs, val_y)

        print(f"  Text-Weight: {alpha}, CLIP-Weight: {1-alpha}")

        # ── 7. Test-Set Prediction ────────────────────────────
        print("\n[7/8] Test-Set predicten...")
        test_text_probs = get_text_probs(text_model, df_test, tokenizer)
        test_clip_probs = clip_zero_shot_scores(df_test)

        ens_probs, pred_matrix, result_tags = predict_ensemble(
            test_text_probs, test_clip_probs, ens_thresholds, alpha
        )

        # ── 8. Submission ─────────────────────────────────────
        print("\n[8/8] Submission erstellen...")
        sub_path = create_submission(result_tags, df_test, version="v1")

        # MLflow Artifact
        try:
            import mlflow
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name="ensemble_xlm_clip"):
                mlflow.log_param("text_weight", alpha)
                mlflow.log_param("clip_weight", 1 - alpha)
                mlflow.log_artifact(str(sub_path))
        except Exception:
            pass

        print("\n" + "=" * 60)
        print("PIPELINE FERTIG!")
        print(f"  Submission: {sub_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rakuten Color Pipeline")
    parser.add_argument("--mode", default="ingest",
                        choices=["full", "ingest", "train", "predict"])
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs: overrides["epochs"] = args.epochs
    if args.lr:     overrides["lr"] = args.lr

    run_pipeline(mode=args.mode, real=args.real, config_overrides=overrides)