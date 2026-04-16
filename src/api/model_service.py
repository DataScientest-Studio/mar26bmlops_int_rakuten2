"""
Model service with automatic fallback chain:
  1. ICE DualEncoder from MLflow Registry (champion alias)
  2. ICE DualEncoder from local checkpoint (fallback)
  3. Keyword matcher (always available)
"""

import time
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    COLOR_LABELS,
    NUM_LABELS,
    MODEL_DIR,
    ICE_CONFIG,
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_CHAMPION_ALIAS,
    IMAGE_SOURCE,
)

COLOR_KEYWORDS: dict[str, list[str]] = {
    "Black": ["ブラック", "黒", "ブラク", "black", "BLACK", "Black"],
    "White": ["ホワイト", "白", "white", "WHITE", "White"],
    "Grey": ["グレー", "グレイ", "灰", "グレ", "gray", "grey", "GRAY", "GREY", "Gray", "Grey"],
    "Navy": ["ネイビー", "紺", "ネービー", "navy", "NAVY", "Navy"],
    "Blue": ["ブルー", "青", "ブル", "blue", "BLUE", "Blue"],
    "Red": ["レッド", "赤", "red", "RED", "Red"],
    "Pink": ["ピンク", "pink", "PINK", "Pink"],
    "Brown": ["ブラウン", "茶", "ダークブラウン", "ライトブラウン", "brown", "BROWN", "Brown"],
    "Beige": ["ベージュ", "beige", "BEIGE", "Beige"],
    "Green": ["グリーン", "緑", "カーキグリーン", "green", "GREEN", "Green"],
    "Khaki": ["カーキ", "khaki", "KHAKI", "Khaki"],
    "Orange": ["オレンジ", "orange", "ORANGE", "Orange"],
    "Yellow": ["イエロー", "黄", "イエロ", "yellow", "YELLOW", "Yellow"],
    "Purple": ["パープル", "紫", "バイオレット", "purple", "PURPLE", "Purple"],
    "Burgundy": ["バーガンディ", "ボルドー", "bordo", "burgundy", "BURGUNDY", "Burgundy"],
    "Gold": ["ゴールド", "金", "gold", "GOLD", "Gold"],
    "Silver": ["シルバー", "銀", "silver", "SILVER", "Silver"],
    "Transparent": ["透明", "クリア", "トランスパレント", "transparent", "clear", "TRANSPARENT", "CLEAR", "Transparent", "Clear"],
    "Multiple Colors": ["マルチ", "マルチカラー", "カラフル", "多色", "multiple", "MULTIPLE", "Multiple"],
}


class ModelService:
    """Manages model lifecycle and inference with automatic fallback."""

    def __init__(self):
        self.model_type: str = "none"
        self.device: str = "cpu"
        self.thresholds: dict[str, float] = {c: 0.5 for c in COLOR_LABELS}
        self.model_source: str = "none"   # "mlflow_registry" | "local" | "keyword"
        self.is_mock: bool = True

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
        self.model_source = "keyword"
        self.is_mock = True
        print("  ! Keyword fallback active (no ML models available)")

    def _try_load_ice(self) -> bool:
        # 1. Try MLflow Registry first
        if self._try_load_from_mlflow():
            return True

        # 2. Fallback: local checkpoint
        if self._try_load_ice_local():
            return True

        # 3. Last resort: keyword matcher
        return False

    # ----------------------------------------------------------------
    # MLflow Registry loading
    # ----------------------------------------------------------------
    def _try_load_from_mlflow(self) -> bool:
        try:
            import mlflow
            import mlflow.pytorch
            from mlflow.tracking import MlflowClient
            import torch

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

            mv = client.get_model_version_by_alias(
                MLFLOW_REGISTERED_MODEL_NAME,
                MLFLOW_CHAMPION_ALIAS,
            )
            model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_CHAMPION_ALIAS}"
            source_run_id = mv.run_id

            print(f"  + Loading champion from MLflow Registry: {model_uri}")
            print(f"    version={mv.version} run_id={source_run_id[:8]}")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # needed for deserialization
            from src.models.train_model_final import ICEModel

            model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            model.to(self.device)
            model.eval()

            artifact_rel_path = f"artifacts/{Path(ICE_CONFIG['mlb_path']).name}"
            mlb_local_path = client.download_artifacts(source_run_id, artifact_rel_path)

            with open(mlb_local_path, "rb") as f:
                self._ice_mlb = pickle.load(f)

            self._setup_ice(model)
            self.model_source = "mlflow_registry"

            print(
                f"  + ICE DualEncoder loaded from MLflow Registry "
                f"({len(self._ice_mlb.classes_)} classes, device={self.device})"
            )
            return True

        except Exception as e:
            print(f"  - MLflow Registry load failed: {e}")
            return False

    # ----------------------------------------------------------------
    # Local checkpoint fallback
    # ----------------------------------------------------------------
    def _try_load_ice_local(self) -> bool:
        ckpt = ICE_CONFIG["checkpoint_path"]
        mlb_p = ICE_CONFIG["mlb_path"]

        if not Path(ckpt).exists() or not Path(mlb_p).exists():
            print(f"  - ICE local: checkpoint not found ({ckpt})")
            return False

        try:
            import torch
            from src.models.train_model_final import (
                DualEncoder,
                ColorClassifier,
                ICEModel,
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            with open(mlb_p, "rb") as f:
                self._ice_mlb = pickle.load(f)

            num_classes = len(self._ice_mlb.classes_)
            self._ice_label_to_idx = {
                label: i for i, label in enumerate(self._ice_mlb.classes_)
            }

            dual_encoder = DualEncoder(
                ICE_CONFIG["text_model_id"],
                ICE_CONFIG["vision_model_id"],
            ).to(self.device)

            classifier = ColorClassifier(
                input_dim=1536,
                num_colors=num_classes,
            ).to(self.device)

            model = ICEModel(dual_encoder, classifier).to(self.device)

            state = torch.load(ckpt, map_location=self.device, weights_only=False)

            if isinstance(state, dict) and "classifier" in state and "dual_encoder" in state:
                classifier.load_state_dict(state["classifier"])
                dual_encoder.load_state_dict(state["dual_encoder"])
            else:
                model.load_state_dict(state)

            model.eval()

            self._setup_ice(model)
            self.model_source = "local"

            print(
                f"  + ICE DualEncoder loaded from local checkpoint "
                f"({num_classes} classes, device={self.device})"
            )
            return True

        except Exception as e:
            print(f"  - ICE local error: {e}")
            return False

    def _setup_ice(self, model):
        """Common setup after model is loaded (registry or local)."""
        from transformers import AutoTokenizer, CLIPImageProcessor

        self._ice_model = model
        self._ice_label_to_idx = {
            label: i for i, label in enumerate(self._ice_mlb.classes_)
        }
        self._ice_tokenizer = AutoTokenizer.from_pretrained(ICE_CONFIG["text_model_id"])
        self._ice_image_processor = CLIPImageProcessor.from_pretrained(
            ICE_CONFIG["vision_model_id"]
        )
        self.model_type = "ice_dual_encoder"
        self.is_mock = False
        self.thresholds = {
            c: ICE_CONFIG.get("val_threshold", 0.5) for c in COLOR_LABELS
        }

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    def predict(
        self,
        item_name: str,
        item_caption: str = "",
        image_path: Optional[str] = None,
    ) -> dict:
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
            "model_source": self.model_source,
            "inference_ms": round(elapsed_ms, 2),
        }

    def _predict_ice(self, item_name, item_caption, image_path=None):
        import torch

        text = f"{item_name} {item_caption}"
        text_enc = self._ice_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=ICE_CONFIG.get("max_len", 128),
            truncation=True,
        )

        image_arr = None
        if image_path:
            try:
                from src.models.train_model_final import load_image_as_rgb_array

                image_arr = load_image_as_rgb_array(
                    image_file_name=Path(image_path).name,
                    image_source=ICE_CONFIG.get("image_source", "local"),
                    img_dir=str(Path(image_path).parent),
                    minio_bucket=ICE_CONFIG.get("minio_bucket_images"),
                    minio_prefix=ICE_CONFIG.get("minio_image_prefix", ""),
                )
            except Exception:
                image_arr = None

        if image_arr is None:
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        img_enc = self._ice_image_processor(
            images=image_arr,
            return_tensors="pt",
            input_data_format="channels_last",
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
                    max_score = max(
                        max_score,
                        min(0.95, 0.6 + 0.15 * count) * pos_factor,
                    )
            if max_score == 0.0:
                max_score = np.random.uniform(0.01, 0.08)
            scores[color] = round(max_score, 4)
        return scores

    def predict_batch(self, items: list[dict]) -> list[dict]:
        return [
            self.predict(
                it["item_name"],
                it.get("item_caption", ""),
                it.get("image_path"),
            )
            for it in items
        ]

    def get_info(self) -> dict:
        info = {
            "model_type": self.model_type,
            "model_source": self.model_source,
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