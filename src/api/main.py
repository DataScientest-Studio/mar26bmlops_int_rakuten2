"""
Rakuten Color Extraction API.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

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


def model_dep() -> ModelService:
    return get_model_service()


def _get_db():
    try:
        from src.db import get_conn, get_db_summary, get_label_distribution, placeholder
        return get_conn, get_db_summary, get_label_distribution, placeholder
    except Exception:
        return None, None, None, lambda: "?"


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
    if image is not None:
        suffix = Path(image.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            image_path = tmp.name
    try:
        result = service.predict(item_name, item_caption, image_path)
    finally:
        if image_path:
            Path(image_path).unlink(missing_ok=True)
    return PredictionResponse(
        predicted_colors=result["predicted"],
        all_scores=_build_scores(result),
        model_type=result["model_type"],
        inference_ms=result["inference_ms"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest, service: ModelService = Depends(model_dep)):
    import time
    start = time.perf_counter()
    predictions = []
    for item in request.items:
        result = service.predict(item.item_name, item.item_caption, item.image_path)
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
    get_conn, _, _, ph = _get_db()
    if get_conn is None:
        raise HTTPException(503, "Database unavailable")
    p = ph()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT image_file_name, item_name, item_caption FROM products WHERE id = {p}",
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
    get_conn, _, _, ph = _get_db()
    if get_conn is None:
        raise HTTPException(503, "Database unavailable")
    p = ph()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, split, image_file_name, item_name, item_caption FROM products WHERE id = {p}",
            (product_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"Product {product_id} not found")
        cur.execute(f"SELECT color_tag FROM labels WHERE product_id = {p}", (product_id,))
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
    get_conn, _, _, ph = _get_db()
    if get_conn is None:
        raise HTTPException(503, "Database unavailable")
    p = ph()
    with get_conn() as conn:
        cur = conn.cursor()
        if split:
            cur.execute(
                f"SELECT id, split, image_file_name, item_name, item_caption "
                f"FROM products WHERE split = {p} ORDER BY id LIMIT {p} OFFSET {p}",
                (split, limit, offset),
            )
        else:
            cur.execute(
                f"SELECT id, split, image_file_name, item_name, item_caption "
                f"FROM products ORDER BY id LIMIT {p} OFFSET {p}",
                (limit, offset),
            )
        products = []
        for row in cur.fetchall():
            cur2 = conn.cursor()
            cur2.execute(f"SELECT color_tag FROM labels WHERE product_id = {p}", (row[0],))
            products.append(ProductResponse(
                id=row[0], split=row[1], image_file=row[2],
                item_name=row[3], item_caption=row[4],
                color_labels=[r[0] for r in cur2.fetchall()],
            ))
    return products


@app.get("/labels", response_model=list[LabelDistribution], tags=["Database"])
def get_labels(split: Optional[str] = Query(None)):
    _, _, get_label_dist, _ = _get_db()
    if get_label_dist is None:
        raise HTTPException(503, "Database unavailable")
    try:
        return [LabelDistribution(color_tag=t, count=c) for t, c in get_label_dist(split=split)]
    except Exception:
        raise HTTPException(503, "Database not initialized")


@app.get("/db/summary", response_model=DBSummaryResponse, tags=["Database"])
def db_summary():
    _, get_summary, _, _ = _get_db()
    if get_summary is None:
        raise HTTPException(503, "Database unavailable")
    try:
        return DBSummaryResponse(**get_summary())
    except Exception:
        raise HTTPException(503, "Database not initialized")


# -- System -------------------------------------------------------------

@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info(service: ModelService = Depends(model_dep)):
    return ModelInfoResponse(**service.get_info())


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(service: ModelService = Depends(model_dep)):
    db_ok = False
    try:
        get_conn, _, _, _ = _get_db()
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