# src/monitoring/drift.py
# Drift detection: compares training vs. val label distributions using Evidently
# Outputs: HTML report + JSON metrics for Prometheus scraping

# start hrml File in Chrome: 
# cmd.exe /c start chrome "$(wslpath -w reports/drift_report.html)"

import json
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS

# -- Paths (must match dvc.yaml predict outputs) --
TRAIN_PRED_PATH = Path("reports/y_pred_train.csv")
VAL_PRED_PATH   = Path("reports/y_pred_val.csv")
REPORT_PATH     = Path("reports/drift_report.html")
METRICS_PATH    = Path("reports/drift_metrics.json")  # consumed by Prometheus exporter


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions CSV and return one-hot encoded label DataFrame."""
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        # color_tags stored as stringified list: "['Red', 'Blue']"
        tags = eval(row["color_tags"]) if isinstance(row["color_tags"], str) else []
        entry = {label: int(label in tags) for label in COLOR_LABELS}
        rows.append(entry)
    return pd.DataFrame(rows)


def compute_label_stats(df: pd.DataFrame) -> dict:
    """Return per-label prevalence (fraction of samples with label = 1)."""
    return {col: float(df[col].mean()) for col in df.columns}


def run_drift():
    print("=== Drift Detection ===")

    if not TRAIN_PRED_PATH.exists() or not VAL_PRED_PATH.exists():
        print("Prediction files missing — skipping drift.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("<html><body>Prediction files missing.</body></html>")
        METRICS_PATH.write_text(json.dumps({"error": "prediction files missing"}))
        return

    print("Loading predictions...")
    train_df = load_predictions(TRAIN_PRED_PATH)
    val_df   = load_predictions(VAL_PRED_PATH)

    # -- Compute label drift metrics manually --
    train_stats = compute_label_stats(train_df)
    val_stats   = compute_label_stats(val_df)

    drift_metrics = {}
    for label in COLOR_LABELS:
        train_p = train_stats.get(label, 0.0)
        val_p   = val_stats.get(label, 0.0)
        drift   = abs(val_p - train_p)
        drift_metrics[label] = {
            "train_prevalence": round(train_p, 4),
            "val_prevalence":   round(val_p, 4),
            "drift":            round(drift, 4),
            "drifted":          drift > 0.05,  # threshold: 5% absolute shift
        }

    # -- Overall drift summary --
    n_drifted = sum(1 for v in drift_metrics.values() if v["drifted"])
    summary = {
        "n_labels":  len(COLOR_LABELS),
        "n_drifted": n_drifted,
        "drift_share": round(n_drifted / len(COLOR_LABELS), 4),
        "max_drift":   round(max(v["drift"] for v in drift_metrics.values()), 4),
        "mean_drift":  round(sum(v["drift"] for v in drift_metrics.values()) / len(COLOR_LABELS), 4),
        "labels":      drift_metrics,
    }

    # -- Save JSON for Prometheus exporter --
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Drift metrics saved to {METRICS_PATH}")
    print(f"  {n_drifted}/{len(COLOR_LABELS)} labels drifted (>{5}% threshold)")

    # -- Evidently HTML report --
    try:
        from evidently.report import Report
        from evidently.metrics import DataDriftTable, DatasetDriftMetric

        # from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

        print("Running Evidently report...")
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        report.run(reference_data=train_df, current_data=val_df)
        report.save_html(str(REPORT_PATH))
        print(f"Evidently HTML report saved to {REPORT_PATH}")

        # -- Extract Evidently JSON for richer metrics --
        result = report.as_dict()
        evidently_drift = result["metrics"][0]["result"]
        summary["evidently"] = {
            "dataset_drift":    evidently_drift.get("dataset_drift", False),
            "share_drifted":    evidently_drift.get("share_of_drifted_columns", 0.0),
            "n_drifted_cols":   evidently_drift.get("number_of_drifted_columns", 0),
        }
        METRICS_PATH.write_text(json.dumps(summary, indent=2))

    except ImportError:
        print("Evidently not installed — HTML report skipped, JSON metrics still saved.")
        REPORT_PATH.write_text(
            "<html><body><h2>Evidently not installed</h2>"
            "<p>JSON metrics available at reports/drift_metrics.json</p></body></html>"
        )
    except Exception as e:
        print(f"Evidently report failed: {e} — JSON metrics still saved.")

    # -- Print summary table --
    print("\n  Label Drift Summary:")
    print(f"  {'Label':<20} {'Train':>8} {'Val':>8} {'Drift':>8} {'Alert':>6}")
    print("  " + "-" * 55)
    for label, m in sorted(drift_metrics.items(), key=lambda x: -x[1]["drift"]):
        alert = "ATTENTION" if m["drifted"] else ""
        print(f"  {label:<20} {m['train_prevalence']:>8.3f} {m['val_prevalence']:>8.3f} {m['drift']:>8.3f} {alert:>6}")


if __name__ == "__main__":
    run_drift()
