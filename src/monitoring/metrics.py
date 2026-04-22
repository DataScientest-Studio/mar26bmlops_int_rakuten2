# src/monitoring/metrics.py
# Prometheus gauges for model performance + data drift.
#
# NOTE: request counters + latency histogram already live in src/api/main.py
# (PrometheusMiddleware). Do NOT add them again here — would cause duplicate
# metric names and conflicting label schemas.
#
# This module only exposes:
#   * MODEL_F1_MICRO / MODEL_F1_MACRO / MODEL_VERSION (updated on reload)
#   * DRIFT_SHARE / DRIFT_MAX / DRIFT_MEAN / DRIFT_N_LABELS (scrape-time)
#   * Per-label drift gauges (created dynamically from drift_metrics.json)

import json
from pathlib import Path

from prometheus_client import Gauge

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_CHAMPION_ALIAS,
)

# -- Model gauges ---------------------------------------------------------

MODEL_F1_MICRO = Gauge(
    "rakuten_model_val_f1_micro",
    "Validation micro-F1 of the current champion model",
)
MODEL_F1_MACRO = Gauge(
    "rakuten_model_val_f1_macro",
    "Validation macro-F1 of the current champion model",
)
MODEL_VERSION = Gauge(
    "rakuten_model_version",
    "Registry version number of the current champion model",
)

# -- Drift gauges ---------------------------------------------------------

DRIFT_SHARE = Gauge(
    "rakuten_drift_share_drifted",
    "Share of label columns flagged as drifted (0..1)",
)
DRIFT_MAX = Gauge(
    "rakuten_drift_max",
    "Maximum per-label drift value observed",
)
DRIFT_MEAN = Gauge(
    "rakuten_drift_mean",
    "Mean per-label drift value observed",
)
DRIFT_N_LABELS = Gauge(
    "rakuten_drift_n_drifted",
    "Number of labels flagged as drifted",
)

# Dynamically-created per-label gauges (populated on first drift update)
_label_drift_gauges: dict[str, Gauge] = {}

DRIFT_METRICS_PATH = Path("reports/drift_metrics.json")


def _get_label_gauge(label: str) -> Gauge:
    """Return (creating if needed) a Gauge for a specific color label."""
    safe = label.lower().replace(" ", "_")
    if safe not in _label_drift_gauges:
        _label_drift_gauges[safe] = Gauge(
            f"rakuten_drift_label_{safe}",
            f"Drift for label '{label}' (train vs val prevalence)",
        )
    return _label_drift_gauges[safe]


# -- Update functions -----------------------------------------------------

def update_drift_metrics() -> None:
    """Reload drift_metrics.json and push values into the gauges.

    Called on every Prometheus scrape so the file stays authoritative —
    the drift pipeline writes the JSON, API reads it without coupling.
    Silent if the file does not exist yet (e.g. before first drift run).
    """
    if not DRIFT_METRICS_PATH.exists():
        return
    try:
        data = json.loads(DRIFT_METRICS_PATH.read_text())
        DRIFT_SHARE.set(data.get("drift_share", 0))
        DRIFT_MAX.set(data.get("max_drift", 0))
        DRIFT_MEAN.set(data.get("mean_drift", 0))
        DRIFT_N_LABELS.set(data.get("n_drifted", 0))
        for label, m in data.get("labels", {}).items():
            _get_label_gauge(label).set(m.get("drift", 0))
    except Exception as e:
        # Don't crash the scrape on malformed JSON — just log and move on
        print(f"[metrics] Failed to load drift metrics: {e}")


def update_model_metrics(f1_micro: float, f1_macro: float, version: int) -> None:
    """Manual setter — call after training/promotion if you already have
    the values in memory (avoids an extra MLflow roundtrip)."""
    MODEL_F1_MICRO.set(f1_micro)
    MODEL_F1_MACRO.set(f1_macro)
    MODEL_VERSION.set(version)


def update_model_metrics_from_mlflow() -> None:
    """Fetch current champion version + F1 scores from MLflow Registry
    and update the gauges accordingly.

    Called on API startup and after /admin/reload so Grafana always shows
    the metrics of the model currently loaded in memory.

    Silently degrades if MLflow is unreachable — gauges stay at previous
    values (or 0 on first start).
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        # Resolve champion alias -> concrete version
        version = client.get_model_version_by_alias(
            MLFLOW_REGISTERED_MODEL_NAME,
            MLFLOW_CHAMPION_ALIAS,
        )
        run_id = version.run_id

        # Pull the run's logged metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics

        # Prefer "best_*" (final summary) over last-epoch values
        f1_micro = metrics.get("best_val_f1_micro") or metrics.get("val_f1_micro", 0.0)
        f1_macro = metrics.get("best_val_f1_macro") or metrics.get("val_f1_macro", 0.0)

        MODEL_F1_MICRO.set(float(f1_micro))
        MODEL_F1_MACRO.set(float(f1_macro))
        MODEL_VERSION.set(int(version.version))

        print(
            f"[metrics] Champion v{version.version}: "
            f"f1_micro={f1_micro:.4f}, f1_macro={f1_macro:.4f}"
        )
    except Exception as e:
        print(f"[metrics] Could not update model metrics from MLflow: {e}")