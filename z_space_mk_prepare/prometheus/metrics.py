# src/monitoring/metrics.py
# Prometheus metrics endpoint — mount into FastAPI app
# Exposes: model performance, drift, request counts

import json
import time
from pathlib import Path
from typing import Optional

from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST,
    multiprocess, REGISTRY,
)
from fastapi import APIRouter, Response

# -- Metric definitions --
registry = REGISTRY  # use default global registry

# Request metrics
REQUEST_COUNT = Counter(
    "rakuten_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "rakuten_api_request_duration_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Model metrics (updated from MLflow/drift reports)
MODEL_F1_MICRO = Gauge("rakuten_model_val_f1_micro", "Val micro-F1 of champion model")
MODEL_F1_MACRO = Gauge("rakuten_model_val_f1_macro", "Val macro-F1 of champion model")
MODEL_VERSION  = Gauge("rakuten_model_version",       "Champion model version number")

# Drift metrics (updated from drift_metrics.json)
DRIFT_SHARE    = Gauge("rakuten_drift_share_drifted",  "Share of drifted label columns")
DRIFT_MAX      = Gauge("rakuten_drift_max",             "Max label drift observed")
DRIFT_MEAN     = Gauge("rakuten_drift_mean",            "Mean label drift observed")
DRIFT_N_LABELS = Gauge("rakuten_drift_n_drifted",       "Number of drifted labels")

# Per-label drift gauges (created dynamically)
_label_drift_gauges: dict = {}


def _get_label_gauge(label: str) -> Gauge:
    """Get or create a per-label drift gauge."""
    safe = label.lower().replace(" ", "_")
    if safe not in _label_drift_gauges:
        _label_drift_gauges[safe] = Gauge(
            f"rakuten_drift_label_{safe}",
            f"Drift for label {label}",
        )
    return _label_drift_gauges[safe]


DRIFT_METRICS_PATH = Path("reports/drift_metrics.json")


def update_drift_metrics():
    """Load drift_metrics.json and update Prometheus gauges."""
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
        print(f"[metrics] Failed to load drift metrics: {e}")


def update_model_metrics(f1_micro: float, f1_macro: float, version: int):
    """Call after training/promotion to update model gauges."""
    MODEL_F1_MICRO.set(f1_micro)
    MODEL_F1_MACRO.set(f1_macro)
    MODEL_VERSION.set(version)


# -- FastAPI router --
router = APIRouter()


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    update_drift_metrics()  # refresh from file on each scrape
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# -- Middleware helper (add to FastAPI app) --
class PrometheusMiddleware:
    """ASGI middleware to track request counts and latency."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "unknown")
        start = time.time()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start
            REQUEST_COUNT.labels(endpoint=path, status=str(status_code)).inc()
            REQUEST_LATENCY.labels(endpoint=path).observe(duration)
