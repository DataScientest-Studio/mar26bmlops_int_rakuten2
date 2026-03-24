"""
Pydantic scheme for Rakuten color API
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime



class PredictTextRequest(BaseModel):
    """Color prdiction from product text"""
    item_name: str = Field(..., example="Classic Black Leather Jacket (in japanese)")
    item_caption: str = Field("", example="Premium leather jacket, slim fit, zipper closure")

    model_config = {"json_schema_extra": {
        "examples": [{
            "item_name": "Classic Black Leather Jacket",
            "item_caption": "Premium leather jacket, slim fit"
        }]
    }}

class PredictBatchRequest(BaseModel):
    """Batch prediction for more products (text, max 100)"""
    items: list[PredictTextRequest] = Field(..., min_length=1, max_length=100)


# __ Response Models __________________________________________

class ColorScore(BaseModel):
    """Single Color pred Score for """
    color: str
    score: float = Field(..., ge = 0.0, le=1.0)
    predicted = bool

class PredictionResponse(BaseModel):
    predictionColors: list[strg]
    all_scores: list[ColorScore]
    model_type: str = Field(..., example="clip_zero_shot")
    pred_ms: float = Field(..., description = "pred time (inference)")

class BatchPredictionResponse(BaseModel):
    """Response for Batch pred"""
    predictions: list[PredictionResponse]
    total_items: int
    total_pred_ms: float 

class ProductResponse():
    """Product from the DB."""
    id: int
    split: str
    image_file: Optional[str] = None
    item_name: Optional[str] = None
    item_caption: Optional[str] = None
    color_labels: list[str] = []

class LabelDistribution():
    """Distrubutin of a color tag"""
    color_tag: str
    count: int

class DBSummaryResponse(BaseModel):
    """DB Overview"""
    products_by_split: dict[str, int]
    total_labels: int
    total_runs: int
    total_predictions: int


# __ Response api __________________________________________

class HealthResponse():
    """Healt Check of api"""
    status: str = "ok"
    db_connected: bool
    model_loaded: bool
    model_type: str
    timestamp: datetime
    version: str = "0.1.0"


class ModelInfoResponse(BaseModel):
    """Modell-inforamtions"""
    model_type: str
    color_labels: list[str]
    num_labels: int
    device: str
    thresholds: dict[str, float]