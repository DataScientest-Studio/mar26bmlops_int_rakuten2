"""
Model service with automatic fallback chain:
  1. ICE DualEncoder (text + image)
  2. Keyword matcher (always available)
"""
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS, NUM_LABELS, MODEL_DIR, ICE_CONFIG


COLOR_KEYWORDS: dict[str, list[str]] = {
    "Black":    ["black", "noir", "schwarz", "nero", "negro"],
    "White":    ["white", "blanc", "weiss", "weiß", "bianco", "blanco"],
    "Grey":     ["grey", "gray", "gris", "grau", "grigio"],
    "Navy":     ["navy", "marine", "dunkelblau"],
    "Blue":     ["blue", "bleu", "blau", "blu", "azul"],
    "Red":      ["red", "rouge", "rot", "rosso", "rojo"],
    "Pink":     ["pink", "rose", "rosa"],
    "Brown":    ["brown", "marron", "braun", "marrone"],
    "Beige":    ["beige", "creme", "cream", "ecru", "sand"],
    "Green":    ["green", "vert", "grün", "gruen", "verde"],
    "Khaki":    ["khaki", "kaki", "olive", "oliv"],
    "Orange":   ["orange", "orangé"],
    "Yellow":   ["yellow", "jaune", "gelb", "giallo"],
    "Purple":   ["purple", "violet", "lila", "violett", "viola"],
    "Burgundy": ["burgundy", "bordeaux", "weinrot", "maroon"],
    "Gold":     ["gold", "golden", "doré"],
    "Silver":   ["silver", "silber", "argent"],
    "Transparent": ["transparent", "clear", "durchsichtig", "translucent"],
    "Multiple Colors": ["multicolor", "multi-color", "bunt", "colorful",
                        "rainbow", "multicolore", "mehrfarbig"],
}


class ModelService:
    """Manages model lifecycle and inference with automatic fallback."""

    def __init__(self):
        self.model_type: str = "none"
        self.device: str = "cpu"
        self.thresholds: dict[str, float] = {c: 0.5 for c in COLOR_LABELS}

        self._ice_model = None
        self._ice_tokenizer = None
        self._ice_image_processor = None
        self._ice_mlb = None
        self._ice_label_to_idx: dict[str, int] = {}

        self._load_model()

    def _load_model(self):
        if self._try_load_ice():
            return
        self.model_type = "keyword_fallback"
        self.is_mock = True
        print("  ! Keyword fallback active (no ML models available)")

    def _try_load_ice(self) -> bool:
        ckpt = ICE_CONFIG["checkpoint_path"]
        mlb_p = ICE_CONFIG["mlb_path"]

        if not ckpt.exists() or not mlb_p.exists():
            print(f"  - ICE: checkpoint not found ({ckpt.name})")
            return False

        try:
            import torch
            from src.models.train_model_ice_mk import (
                DualEncoder, ColorClassifier, ICEModel,
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            with open(mlb_p, "rb") as f:
                self._ice_mlb = pickle.load(f)

            num_classes = len(self._ice_mlb.classes_)
            self._ice_label_to_idx = {
                label: i for i, label in enumerate(self._ice_mlb.classes_)
            }

            dual_encoder = DualEncoder(
                ICE_CONFIG["text_model_id"], ICE_CONFIG["vision_model_id"]
            ).to(self.device)
            classifier = ColorClassifier(input_dim=1536, num_colors=num_classes).to(self.device)
            model = ICEModel(dual_encoder, classifier).to(self.device)

            state = torch.load(ckpt, map_location=self.device, weights_only=False)

            # Handle both formats: full ICEModel state_dict or split checkpoint
            if isinstance(state, dict) and "classifier" in state and "dual_encoder" in state:
                classifier.load_state_dict(state["classifier"])
                dual_encoder.load_state_dict(state["dual_encoder"])
            else:
                model.load_state_dict(state)

            model.eval()
            self._ice_model = model

            from transformers import AutoTokenizer, CLIPImageProcessor
            self._ice_tokenizer = AutoTokenizer.from_pretrained(ICE_CONFIG["text_model_id"])
            self._ice_image_processor = CLIPImageProcessor.from_pretrained(ICE_CONFIG["vision_model_id"])

            self.model_type = "ice_dual_encoder"
            self.is_mock = False
            self.thresholds = {c: ICE_CONFIG.get("val_threshold", 0.5) for c in COLOR_LABELS}

            print(f"  + ICE DualEncoder loaded ({num_classes} classes, device={self.device})")
            return True

        except Exception as e:
            print(f"  - ICE error: {e}")
            return False

    def predict(self, item_name: str, item_caption: str = "",
                image_path: Optional[str] = None) -> dict:
        start = time.perf_counter()

        if self.model_type == "ice_dual_encoder":
            scores = self._predict_ice(item_name, item_caption, image_path)
        else:
            scores = self._predict_keywords(item_name, item_caption)

        predicted = [c for c, s in scores.items() if s >= self.thresholds.get(c, 0.5)]
        if not predicted:
            predicted = [max(scores, key=scores.get)]

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "scores": scores,
            "predicted": predicted,
            "model_type": self.model_type,
            "inference_ms": round(elapsed_ms, 2),
        }

    def _predict_ice(self, item_name, item_caption, image_path=None):
        import torch

        text = f"{item_name} {item_caption}"
        text_enc = self._ice_tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=ICE_CONFIG.get("max_len", 128), truncation=True,
        )

        if image_path and Path(image_path).exists():
            try:
                from src.models.train_model_ice_mk import load_image_as_rgb_array
                image_arr = load_image_as_rgb_array(image_path)
            except Exception:
                image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)
        else:
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        img_enc = self._ice_image_processor(
            images=image_arr, return_tensors="pt", input_data_format="channels_last",
        )

        with torch.no_grad():
            ids = text_enc["input_ids"].to(self.device)
            mask = text_enc["attention_mask"].to(self.device)
            px = img_enc["pixel_values"].to(self.device)
            logits = self._ice_model(ids, mask, px)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        scores = {}
        for color in COLOR_LABELS:
            if color in self._ice_label_to_idx:
                scores[color] = round(float(probs[self._ice_label_to_idx[color]]), 4)
            else:
                scores[color] = 0.0
        return scores

    def _predict_keywords(self, item_name, item_caption):
        text = f"{item_name} {item_caption}".lower()
        scores = {}
        for color, keywords in COLOR_KEYWORDS.items():
            max_score = 0.0
            for kw in keywords:
                if kw in text:
                    count = text.count(kw)
                    pos = text.find(kw)
                    pos_factor = max(0.5, 1.0 - pos / max(len(text), 1))
                    max_score = max(max_score, min(0.95, 0.6 + 0.15 * count) * pos_factor)
            if max_score == 0.0:
                max_score = np.random.uniform(0.01, 0.08)
            scores[color] = round(max_score, 4)
        return scores

    def predict_batch(self, items: list[dict]) -> list[dict]:
        return [
            self.predict(it["item_name"], it.get("item_caption", ""), it.get("image_path"))
            for it in items
        ]

    def get_info(self) -> dict:
        info = {
            "model_type": self.model_type,
            "color_labels": COLOR_LABELS,
            "num_labels": NUM_LABELS,
            "device": self.device,
            "is_mock": self.is_mock,
            "thresholds": self.thresholds,
        }
        if self._ice_mlb is not None:
            info["ice_classes"] = list(self._ice_mlb.classes_)
        return info


_service: Optional[ModelService] = None

def get_model_service() -> ModelService:
    global _service
    if _service is None:
        print("\n-> Loading models...")
        _service = ModelService()
    return _service