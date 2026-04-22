# src/monitoring/drift.py
# Drift detection: two independent reports + JSON metrics for Prometheus.
#
#   1. data_drift_report.html  — dark vs light color group drift (DB-based)
#   2. label_drift_report.html — train vs val label distribution drift (CSV-based)
#   3. drift_metrics.json      — machine-readable summary for Prometheus gauges
#
# Open HTML from WSL:
#   cmd.exe /c start chrome "$(wslpath -w reports/data_drift_report.html)"

import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
METRICS_JSON = REPORTS_DIR / "drift_metrics.json"


# =========================================================================
# Report 1: DB-based dark vs light color group drift
# =========================================================================

DARK_COLORS = {"Black", "Navy", "Brown", "Grey"}
LIGHT_COLORS = {"Purple", "Orange", "Pink", "Yellow"}


def run_drift_color_groups():
    """Compare dark-color products vs light-color products using DB data.

    Demo report showing Evidently's drift detection on two real sub-populations.
    Outputs HTML for manual inspection.
    """
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
    print(f"Color group drift report -> {REPORTS_DIR / 'data_drift_report.html'}")


# =========================================================================
# Report 2: CSV-based train vs val label drift
# =========================================================================

TRAIN_PRED_PATH = Path("reports/y_pred_train.csv")
VAL_PRED_PATH = Path("reports/y_pred_val.csv")


def _load_predictions_onehot(path: Path) -> pd.DataFrame:
    """Load a prediction CSV and return a one-hot DataFrame (one row per product,
    one column per COLOR_LABEL)."""
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        tags = eval(row["color_tags"]) if isinstance(row["color_tags"], str) else []
        rows.append({label: int(label in tags) for label in COLOR_LABELS})
    return pd.DataFrame(rows)


def _compute_drift_summary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Per-label prevalence comparison + overall summary for Prometheus."""
    train_stats = {col: float(train_df[col].mean()) for col in train_df.columns}
    val_stats = {col: float(val_df[col].mean()) for col in val_df.columns}

    per_label = {}
    for label in COLOR_LABELS:
        tp = train_stats.get(label, 0.0)
        vp = val_stats.get(label, 0.0)
        d = abs(vp - tp)
        per_label[label] = {
            "train_prevalence": round(tp, 4),
            "val_prevalence":   round(vp, 4),
            "drift":            round(d, 4),
            "drifted":          d > 0.05,  # 5% absolute shift threshold
        }

    n_drifted = sum(1 for v in per_label.values() if v["drifted"])
    return {
        "n_labels":    len(COLOR_LABELS),
        "n_drifted":   n_drifted,
        "drift_share": round(n_drifted / len(COLOR_LABELS), 4),
        "max_drift":   round(max(v["drift"] for v in per_label.values()), 4),
        "mean_drift":  round(sum(v["drift"] for v in per_label.values()) / len(COLOR_LABELS), 4),
        "labels":      per_label,
    }


def run_drift_train_val():
    """Compare train vs val label distributions.

    Produces:
      * reports/label_drift_report.html  (Evidently visual)
      * reports/drift_metrics.json       (Prometheus-scraped by metrics.py)
    """
    if not TRAIN_PRED_PATH.exists() or not VAL_PRED_PATH.exists():
        print("Prediction files missing — skipping label drift report.")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_JSON.write_text(json.dumps({"error": "prediction files missing"}))
        return

    train_df = _load_predictions_onehot(TRAIN_PRED_PATH)
    val_df = _load_predictions_onehot(VAL_PRED_PATH)

    # -- Always produce the JSON summary for Prometheus --
    summary = _compute_drift_summary(train_df, val_df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Drift metrics -> {METRICS_JSON}  "
          f"({summary['n_drifted']}/{summary['n_labels']} labels drifted)")

    # -- Additionally: Evidently HTML for visual inspection --
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(current_data=val_df, reference_data=train_df)
        snapshot.save_html(str(REPORTS_DIR / "label_drift_report.html"))
        print(f"Label drift report -> {REPORTS_DIR / 'label_drift_report.html'}")
    except ImportError:
        print("Evidently not installed — HTML skipped, JSON metrics still saved.")
    except Exception as e:
        print(f"Evidently HTML failed ({e}) — JSON metrics still saved.")

    # -- Pretty-print summary table --
    print("\n  Label Drift Summary:")
    print(f"  {'Label':<20} {'Train':>8} {'Val':>8} {'Drift':>8} {'Alert':>6}")
    print("  " + "-" * 55)
    for label, m in sorted(summary["labels"].items(), key=lambda x: -x[1]["drift"]):
        alert = "!" if m["drifted"] else ""
        print(f"  {label:<20} {m['train_prevalence']:>8.3f} "
              f"{m['val_prevalence']:>8.3f} {m['drift']:>8.3f} {alert:>6}")


if __name__ == "__main__":
    run_drift_color_groups()
    run_drift_train_val()