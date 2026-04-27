# src/monitoring/drift.py
# Drift detection: two independent reports
#   1. data_drift_report.html  — dark vs light color group drift (DB-based)
#   2. label_drift_report.html — train vs val label distribution drift (CSV-based)

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# -- Report 1: DB-based dark vs light color drift ------------------

DARK_COLORS = {"Black", "Navy", "Brown", "Grey"}
LIGHT_COLORS = {"Purple", "Orange", "Pink", "Yellow"}


def run_drift_color_groups():
    """Compare dark-color products vs light-color products using DB data."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("Evidently not installed — skipping color group drift report.")
        return

    from src.db import get_split_data

    df_x, df_y = get_split_data(split="val")
    df = df_x.merge(df_y, on="product_id")
    df["color_list"] = df["color_tags"].str.split(",")

    ref_df = df[df["color_list"].apply(lambda tags: bool(DARK_COLORS & set(tags)))]
    curr_df = df[df["color_list"].apply(lambda tags: bool(LIGHT_COLORS & set(tags)))]

    all_colors = sorted(DARK_COLORS | LIGHT_COLORS)
    ref_encoded = pd.DataFrame(
        [{c: int(c in tags) for c in all_colors} for tags in ref_df["color_list"]]
    )
    curr_encoded = pd.DataFrame(
        [{c: int(c in tags) for c in all_colors} for tags in curr_df["color_list"]]
    )

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=curr_encoded, reference_data=ref_encoded)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_drift_report.json", "w") as f:
        f.write(snapshot.json())
    snapshot.save_html(str(REPORTS_DIR / "data_drift_report.html"))
    print(f"Color group drift report saved to {REPORTS_DIR / 'data_drift_report.html'}")


# -- Report 2: CSV-based train vs val label drift ----------------

TRAIN_PRED_PATH = Path("reports/y_pred_train.csv")
VAL_PRED_PATH = Path("reports/y_pred_val.csv")


def _load_label_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        tags = eval(row["color_tags"]) if isinstance(row["color_tags"], str) else []
        for label in COLOR_LABELS:
            rows.append({"color": label, "present": int(label in tags)})
    return pd.DataFrame(rows)


def run_drift_train_val():
    """Compare train vs val label distributions using prediction CSV files."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("Evidently not installed — skipping label drift report.")
        return

    if not TRAIN_PRED_PATH.exists() or not VAL_PRED_PATH.exists():
        print(f"Prediction files missing — skipping label drift report.")
        return

    train_df = _load_label_counts(TRAIN_PRED_PATH)
    val_df = _load_label_counts(VAL_PRED_PATH)

    train_pivot = train_df.groupby(
        train_df.index // len(COLOR_LABELS)
    ).apply(lambda x: pd.Series(x.set_index("color")["present"]))

    val_pivot = val_df.groupby(
        val_df.index // len(COLOR_LABELS)
    ).apply(lambda x: pd.Series(x.set_index("color")["present"]))

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=val_pivot, reference_data=train_pivot)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORTS_DIR / "label_drift_report.html"))
    print(f"Label drift report saved to {REPORTS_DIR / 'label_drift_report.html'}")


if __name__ == "__main__":
    run_drift_color_groups()
    run_drift_train_val()
