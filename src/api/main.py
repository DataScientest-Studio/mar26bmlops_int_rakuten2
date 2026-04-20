"""
Rakuten Color Extraction API.

Endpoints:
    POST /predict              — predict colors from text (+optional image)
    POST /predict/batch        — batch prediction
    GET  /predict/product/{id} — predict from DB product
    GET  /products/{id}        — load product from DB
    GET  /products             — list products (with filter)
    GET  /labels               — label distribution
    GET  /db/summary           — database overview
    GET  /model/info           — model information
    GET  /health               — health check

Start:
    uvicorn src.api.main:app --reload --port 8000
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from prometheus_client import Counter, Histogram, make_asgi_app


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.schemas import (
    PredictRequest, BatchPredictRequest,
    PredictionResponse, BatchPredictionResponse,
    ColorScore, ProductResponse, LabelDistribution,
    DBSummaryResponse, HealthResponse, ModelInfoResponse,
)
from src.api.model_service import get_model_service, ModelService
from src.config import COLOR_LABELS

app = FastAPI(
    title="Rakuten Color Extraction API",
    description="Color classification from product text and images.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

color_predictions_counter = Counter(
    "rakuten_color_predictions_total",
    "Total number of times each color was predicted",
    ["color"],
)

requests_counter = Counter(
    "rakuten_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

requests_latency = Histogram(
    "rakuten_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        if endpoint != "/metrics":  # avoid self-scrape noise
            requests_counter.labels(request.method, endpoint, response.status_code).inc()
            requests_latency.labels(request.method, endpoint).observe(duration)

        return response

app.add_middleware(PrometheusMiddleware)

app.mount("/metrics", make_asgi_app())


def model_dep() -> ModelService:
    return get_model_service()


def _get_db():
    try:
        from src.db import get_conn, get_db_summary, get_label_distribution
        return get_conn, get_db_summary, get_label_distribution
    except Exception:
        return None, None, None


def _build_scores(result: dict) -> list[ColorScore]:
    scores = [
        ColorScore(
            color=c,
            score=round(result["scores"][c], 4),
            predicted=(c in result["predicted"]),
        )
        for c in COLOR_LABELS
    ]
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores


# -- Prediction ---------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_colors(request: PredictRequest, service: ModelService = Depends(model_dep)):
    result = service.predict(request.item_name, request.item_caption, request.image_path)
    for color in result["predicted"]:
        color_predictions_counter.labels(color=color).inc()
    return PredictionResponse(
        predicted_colors=result["predicted"],
        all_scores=_build_scores(result),
        model_type=result["model_type"],
        inference_ms=result["inference_ms"],
    )



@app.post("/predict/upload", response_model=PredictionResponse, tags=["Prediction"])
async def predict_with_upload(
    item_name: str,
    item_caption: str,
    image: UploadFile = File(None),
    service: ModelService = Depends(model_dep),
):
    import tempfile, shutil

    image_path = None

    # Bild temporär speichern falls mitgegeben
    if image is not None:
        suffix = Path(image.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            image_path = tmp.name

    try:
        result = service.predict(item_name, item_caption, image_path)
    finally:
        # Temp-Datei aufräumen
        if image_path:
            Path(image_path).unlink(missing_ok=True)

    for color in result["predicted"]:
        color_predictions_counter.labels(color=color).inc()
    return PredictionResponse(
        predicted_colors=result["predicted"],
        all_scores=_build_scores(result),
        model_type=result["model_type"],
        inference_ms=result["inference_ms"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest, service: ModelService = Depends(model_dep)):
    start = time.perf_counter()
    predictions = []
    for item in request.items:
        result = service.predict(item.item_name, item.item_caption, item.image_path)
        for color in result["predicted"]:
            color_predictions_counter.labels(color=color).inc()
        predictions.append(PredictionResponse(
            predicted_colors=result["predicted"],
            all_scores=_build_scores(result),
            model_type=result["model_type"],
            inference_ms=result["inference_ms"],
        ))
    total_ms = (time.perf_counter() - start) * 1000
    return BatchPredictionResponse(
        predictions=predictions,
        total_items=len(predictions),
        total_inference_ms=round(total_ms, 2),
    )


@app.get("/predict/product/{product_id}", response_model=PredictionResponse, tags=["Prediction"])
def predict_from_product(product_id: int, service: ModelService = Depends(model_dep)):
    get_conn, _, _ = _get_db()
    if get_conn is None:
        raise HTTPException(503, "Database unavailable")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT image_file_name, item_name, item_caption FROM products WHERE id = ?",
            (product_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"Product {product_id} not found")
    image_file, item_name, item_caption = row
    image_path = None
    if image_file:
        from src.config import DATA_DIR
        candidate = DATA_DIR / "images" / image_file
        if candidate.exists():
            image_path = str(candidate)
    result = service.predict(item_name or "", item_caption or "", image_path)
    return PredictionResponse(
        predicted_colors=result["predicted"],
        all_scores=_build_scores(result),
        model_type=result["model_type"],
        inference_ms=result["inference_ms"],
    )


# -- Database -----------------------------------------------------------

@app.get("/products/{product_id}", response_model=ProductResponse, tags=["Database"])
def get_product(product_id: int):
    get_conn, _, _ = _get_db()
    if get_conn is None:
        raise HTTPException(503, "Database unavailable")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, split, image_file_name, item_name, item_caption "
            "FROM products WHERE id = ?", (product_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"Product {product_id} not found")
        cur.execute("SELECT color_tag FROM labels WHERE product_id = ?", (product_id,))
        labels = [r[0] for r in cur.fetchall()]
    return ProductResponse(
        id=row[0], split=row[1], image_file=row[2],
        item_name=row[3], item_caption=row[4], color_labels=labels,
    )


@app.get("/products", response_model=list[ProductResponse], tags=["Database"])
def list_products(
    split: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    get_conn, _, _ = _get_db()
    if get_conn is None:
        raise HTTPException(503, "Database unavailable")
    with get_conn() as conn:
        cur = conn.cursor()
        if split:
            cur.execute(
                "SELECT id, split, image_file_name, item_name, item_caption "
                "FROM products WHERE split = ? ORDER BY id LIMIT ? OFFSET ?",
                (split, limit, offset),
            )
        else:
            cur.execute(
                "SELECT id, split, image_file_name, item_name, item_caption "
                "FROM products ORDER BY id LIMIT ? OFFSET ?",
                (limit, offset),
            )
        products = []
        for row in cur.fetchall():
            cur2 = conn.cursor()
            cur2.execute("SELECT color_tag FROM labels WHERE product_id = ?", (row[0],))
            products.append(ProductResponse(
                id=row[0], split=row[1], image_file=row[2],
                item_name=row[3], item_caption=row[4],
                color_labels=[r[0] for r in cur2.fetchall()],
            ))
    return products


@app.get("/labels", response_model=list[LabelDistribution], tags=["Database"])
def get_labels(split: Optional[str] = Query(None)):
    _, _, get_label_dist = _get_db()
    if get_label_dist is None:
        raise HTTPException(503, "Database unavailable")
    try:
        return [LabelDistribution(color_tag=t, count=c) for t, c in get_label_dist(split=split)]
    except Exception:
        raise HTTPException(503, "Database not initialized — run: python -m src.pipeline --mode ingest")


@app.get("/db/summary", response_model=DBSummaryResponse, tags=["Database"])
def db_summary():
    _, get_summary, _ = _get_db()
    if get_summary is None:
        raise HTTPException(503, "Database unavailable")
    try:
        return DBSummaryResponse(**get_summary())
    except Exception:
        raise HTTPException(503, "Database not initialized — run: python -m src.pipeline --mode ingest")


# -- System -------------------------------------------------------------

@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info(service: ModelService = Depends(model_dep)):
    return ModelInfoResponse(**service.get_info())


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(service: ModelService = Depends(model_dep)):
    db_ok = False
    try:
        get_conn, _, _ = _get_db()
        if get_conn:
            with get_conn() as conn:
                conn.cursor().execute("SELECT 1")
            db_ok = True
    except Exception:
        pass
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db_connected=db_ok,
        model_loaded=not service.is_mock,
        model_type=service.model_type,
        timestamp=datetime.now(timezone.utc),
    )


@app.on_event("startup")
async def startup():
    print("\n" + "=" * 50)
    print("  Rakuten Color Extraction API")
    print("=" * 50)
    svc = get_model_service()
    print(f"  Model: {svc.model_type} | Device: {svc.device} | Mock: {svc.is_mock}")
    print("=" * 50 + "\n")