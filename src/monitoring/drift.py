# src/monitoring/drift.py
# Drift detection: compares training vs. val label distributions using Evidently

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS

# -- Paths --
TRAIN_PRED_PATH = Path("models/y_pred_train.csv")
VAL_PRED_PATH   = Path("reports/y_pred_val.csv")
REPORT_PATH     = Path("reports/drift_report.html")

def load_label_counts(path: Path) -> pd.DataFrame:
    """Load predictions and count label occurrences per color."""
    df = pd.read_csv(path)
    # Parse color_tags column (stored as string list)
    rows = []
    for _, row in df.iterrows():
        tags = eval(row["color_tags"]) if isinstance(row["color_tags"], str) else []
        for label in COLOR_LABELS:
            rows.append({"color": label, "present": int(label in tags)})
    return pd.DataFrame(rows)

def run_drift():
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        print("Evidently not installed — skipping drift report.")
        REPORT_PATH.write_text("<html><body>Evidently not installed.</body></html>")
        return

    print("Loading predictions...")
    if not TRAIN_PRED_PATH.exists() or not VAL_PRED_PATH.exists():
        print("Prediction files missing — skipping drift.")
        REPORT_PATH.write_text("<html><body>Prediction files missing.</body></html>")
        return

    # Build label frequency tables
    train_df = load_label_counts(TRAIN_PRED_PATH)
    val_df   = load_label_counts(VAL_PRED_PATH)

    # Pivot: one row per sample, one col per color
    train_pivot = train_df.groupby(
        train_df.index // len(COLOR_LABELS)
    ).apply(lambda x: pd.Series(x.set_index("color")["present"]))

    val_pivot = val_df.groupby(
        val_df.index // len(COLOR_LABELS)
    ).apply(lambda x: pd.Series(x.set_index("color")["present"]))

    print("Running Evidently drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_pivot, current_data=val_pivot)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    print(f"Drift report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_drift()