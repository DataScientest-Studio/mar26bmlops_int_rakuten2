"""
Prediction und Submission-Erstellung.

Nutzung:
    python -m src.models.predict_model --run_id <mlflow_run_id>
"""
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS, NUM_LABELS, SUBMIT_DIR


def predict_ensemble(text_probs, clip_probs, thresholds, alpha=0.65):
    """
    Late Fusion: gewichtete Kombination von Text- und CLIP-Scores.

    Args:
        text_probs: (n, NUM_LABELS) Text-Modell Wahrscheinlichkeiten
        clip_probs: (n, NUM_LABELS) CLIP-Modell Wahrscheinlichkeiten
        thresholds: Dict {color: threshold}
        alpha: Gewicht fuer Text-Modell (1-alpha fuer CLIP)

    Returns:
        ensemble_probs, pred_matrix, result_tags
    """
    ensemble_probs = alpha * text_probs + (1 - alpha) * clip_probs

    t_arr = np.array([thresholds.get(c, 0.5) for c in COLOR_LABELS])
    pred_matrix = (ensemble_probs >= t_arr).astype(int)

    # Tags extrahieren
    result_tags = []
    for row in pred_matrix:
        tags = [COLOR_LABELS[j] for j, v in enumerate(row) if v]
        if not tags:
            # Fallback: hoechste Wahrscheinlichkeit nehmen
            tags = [COLOR_LABELS[np.argmax(ensemble_probs[row.tolist().index(0) if 0 in row else 0])]]
            tags = [COLOR_LABELS[np.argmax(ensemble_probs[len(result_tags)])]]
        result_tags.append(tags)

    return ensemble_probs, pred_matrix, result_tags


def create_submission(result_tags, df_test, version="v1"):
    """Erstellt Submission-CSV im Rakuten-Format."""
    SUBMIT_DIR.mkdir(exist_ok=True)

    submission = pd.DataFrame({
        'Column1':    range(len(df_test)),
        'color_tags': [str(tags) for tags in result_tags]
    })

    out_path = SUBMIT_DIR / f"ensemble_{version}.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Submission gespeichert: {out_path} ({len(submission)} Zeilen)")
    return out_path


def predict_text_only(text_probs, thresholds):
    """Nur Text-Modell Predictions (ohne CLIP)."""
    t_arr = np.array([thresholds.get(c, 0.5) for c in COLOR_LABELS])
    pred_matrix = (text_probs >= t_arr).astype(int)

    result_tags = []
    for i, row in enumerate(pred_matrix):
        tags = [COLOR_LABELS[j] for j, v in enumerate(row) if v]
        if not tags:
            tags = [COLOR_LABELS[np.argmax(text_probs[i])]]
        result_tags.append(tags)

    return pred_matrix, result_tags
