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

app = FastAPI(
    title="Rakuten Color Extraction API",

    description=("Extraction of Productcolors from Imgs and Text.\n"
    "Part of the Mlops pipline: FatsAPI > PostgreSQL > MLFlow > Monitoring"
    ),
    version="0.1.0",
    docs_url="/logs",
    redoc_url="/redoc",
)

# Not finished
