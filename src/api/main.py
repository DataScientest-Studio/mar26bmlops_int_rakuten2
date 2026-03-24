"""
Rakuten Color Extraction API

Endpoints
    GET     /Health

Start:
    uvicorn src.api.main:app --reload --port 8000
"""
import sys
from pathlib import Path
from fastapi import FastAPI

# Projekt-Root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.api.scheme import (
    PredictTextRequest, PredictBatchRequest,
    PredictionResponse, BatchPredictionResponse,
    ColorScore, ProductResponse, LabelDistribution,
    DBSummaryResponse, HealthResponse, ModelInfoResponse,
)



app = FastAPI(
    title="Rakuten Color Extraction API",

    description=("Extraction of Productcolors from Imgs and Text.\n"
    "Part of the Mlops pipline: FatsAPI > PostgreSQL > MLFlow > Monitoring"
    ),
    version="0.1.0",
    docs_url="/logs",
    redoc_url="/redoc",
)





# ═══════════════════════════════════════════════════════════════
# MODEL / SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════


@app.get("/health", response_model= HealthResponse, tags=["System"])
def health(service=):


# ── Standalone ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )