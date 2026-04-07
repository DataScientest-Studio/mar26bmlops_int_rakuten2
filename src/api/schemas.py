from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictRequest(BaseModel):
    item_name: str = Field(..., examples=["Classic Black Leather Jacket"])
    item_caption: str = Field("", examples=["Premium leather, slim fit"])
    image_path: Optional[str] = Field(None, description="Path to product image (for ICE model)")


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class ColorScore(BaseModel):
    color: str
    score: float = Field(..., ge=0.0, le=1.0)
    predicted: bool


class PredictionResponse(BaseModel):
    predicted_colors: list[str]
    all_scores: list[ColorScore]
    model_type: str
    inference_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_items: int
    total_inference_ms: float


class ProductResponse(BaseModel):
    id: int
    split: str
    image_file: Optional[str] = None
    item_name: Optional[str] = None
    item_caption: Optional[str] = None
    color_labels: list[str] = []


class LabelDistribution(BaseModel):
    color_tag: str
    count: int


class DBSummaryResponse(BaseModel):
    products_by_split: dict[str, int]
    total_labels: int
    total_runs: int
    total_predictions: int


class HealthResponse(BaseModel):
    status: str = "ok"
    db_connected: bool
    model_loaded: bool
    model_type: str
    timestamp: datetime
    version: str = "0.2.0"


class ModelInfoResponse(BaseModel):
    model_type: str
    color_labels: list[str]
    num_labels: int
    device: str
    is_mock: bool
    thresholds: dict[str, float]
    ice_classes: Optional[list[str]] = None