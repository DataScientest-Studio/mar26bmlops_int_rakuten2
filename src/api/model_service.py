
"""
Model Service: load models and execute prediction

first XLM-RoBERTa
"""

import time
import re
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS, NUM_LABELS, MODEL_DIR

class ModelService:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type: str = "none"
        self.device: str = "cpu"
        self.threshold: dict[str, float] = {c: 0.5 for c in COLOR_LABELS}
    
    def _load_model(self):
        """prio load model, first XLM-RoBERTa"""
        xlm_path = MODEL_DIR / "xlm_best.pt"
        if xlm_path.exists():
            try:
                import torch
                from src.models.train_model import TextColorClassifier
                from src.config import XLM_CONFIG
                from transformers import AutoTokenizer

                # Dense Head
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = TextColorClassifier(
                    XLM_CONFIG["model_name"],
                    num_labels=NUM_LABELS,
                    dropout=0.0         # for inference no dropout
                ).to(self.device)
                self.model.load_state_dict(
                    torch.load(xlm_path, map_location=self.device, weights_only=True)
                )
                self.model_type = "xlm_roberta_finetuned"

                # Optimizing: load Threshold if available
                thresholds_path = MODEL_DIR / "thresholds.json"
                if thresholds_path.exists():
                    import json
                    self.thresholds = json.loads(thresholds_path.read_text())

                print(f"XLM-RoBERTa loaded: {xlm_path}")
                return
            except Exception as e:
                print(f"XLM-RoBERTa Error: {e}")

    # ── Inference ──────────────────────────────────────────────
    def predict(self, 