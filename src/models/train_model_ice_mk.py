"""
ICE (Image-Caption Ensemble) — DualEncoder training with MLflow tracking,
registry integration, model comparison, and artifact logging.

Architecture:
    Text:   cl-tohoku/bert-base-japanese-v3  -> 768d CLS token
    Vision: openai/clip-vit-base-patch16     -> 768d pooler output
    Head:   concat(768,768) -> 1024 -> 512 -> num_colors (MLP)

Usage:
    python -m src.models.train_model_ice_mk
    python -m src.models.train_model_ice_mk --epochs 10 --batch_size 64
"""

import os
import sys
import json
import copy
import pickle
import hashlib
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModel,
    CLIPImageProcessor,
)

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    ICE_CONFIG,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_CANDIDATE_ALIAS,
)
from src.db import get_split_data, save_run

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True


# ============================================================
# Helpers: hashing / manifests / logging
# ============================================================

def sanitize_params(params: dict) -> dict:
    clean = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            clean[key] = "None"
        else:
            clean[key] = str(value)
    return clean


def json_sha256(payload) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    payload = df.to_json(orient="split", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)


def get_model_version_for_run(client: MlflowClient, model_name: str, run_id: str):
    versions = client.search_model_versions(f"name = '{model_name}'")
    matching = [v for v in versions if v.run_id == run_id]
    if not matching:
        return None
    matching.sort(key=lambda x: int(x.version), reverse=True)
    return matching[0]


def get_version_by_alias(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def set_registered_model_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    client.set_registered_model_alias(model_name, alias, version)


def set_model_version_tags(
    client: MlflowClient,
    model_name: str,
    version: str,
    tags: dict,
) -> None:
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, str(value))


def is_better_model(new_score: float, current_score: float | None) -> bool:
    if current_score is None:
        return True
    return new_score > current_score


def log_json_artifact(filename: str, payload: dict, artifact_path: str = "metadata") -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / filename
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(out), artifact_path=artifact_path)


# ============================================================
# Model Architecture
# ============================================================

class DualEncoder(nn.Module):
    """
    Wraps a text encoder (BERT) and a vision encoder (CLIP ViT).
    All parameters frozen on init; selectively unfrozen via
    unfreeze_encoder_layers().
    """

    def __init__(self, text_model_id: str, vision_model_id: str):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_id)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_id)

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def get_text_features(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS -> (B, 768)

    def get_image_features(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values)
        return out.pooler_output  # (B, 768)

    def unfreeze_encoder_layers(self, num_layers: int = 2):
        for layer in self.text_encoder.encoder.layer[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        if getattr(self.text_encoder, "pooler", None) is not None:
            for p in self.text_encoder.pooler.parameters():
                p.requires_grad = True

        for layer in self.vision_encoder.vision_model.encoder.layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        for p in self.vision_encoder.vision_model.post_layernorm.parameters():
            p.requires_grad = True

        unfrozen = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"  Unfroze last {num_layers} layers: "
            f"{unfrozen:,} / {total:,} params ({100 * unfrozen / total:.1f}%)"
        )


class ColorClassifier(nn.Module):
    """Three-layer MLP: (B, 1536) -> (B, num_colors) logits."""

    def __init__(self, input_dim: int = 1536, num_colors: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_colors),
        )

    def forward(self, img_features, txt_features):
        img_norm = img_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        txt_norm = txt_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        img_features = img_features / img_norm
        txt_features = txt_features / txt_norm
        return self.fc(torch.cat((img_features, txt_features), dim=-1))


class ICEModel(nn.Module):
    """
    Wrapper model so MLflow can log/load a single PyTorch model object.
    """

    def __init__(self, dual_encoder: DualEncoder, classifier: ColorClassifier):
        super().__init__()
        self.dual_encoder = dual_encoder
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, pixel_values):
        img_e = self.dual_encoder.get_image_features(pixel_values)
        txt_e = self.dual_encoder.get_text_features(input_ids, attention_mask)
        return self.classifier(img_e, txt_e)


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float("-inf")
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, score, model, epoch_idx: int):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_idx
            print(f"  [ES] Val micro-F1 improved to {score:.6f}")
        else:
            self.counter += 1
            print(
                f"  [ES] No improvement {self.counter}/{self.patience} "
                f"(best: {self.best_score:.6f})"
            )
            if self.counter >= self.patience:
                print("  [ES] Patience exhausted — stopping.")
                return True
        return False

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(
                f"  [ES] Best weights restored "
                f"(F1={self.best_score:.6f}, epoch={self.best_epoch})"
            )


# ============================================================
# Image Loading
# ============================================================

def load_image_as_rgb_array(path: str) -> np.ndarray:
    """Open image and return guaranteed (H, W, 3) uint8 array."""
    image = Image.open(path).convert("RGB")
    arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[:, :, :3]
    return arr


# ============================================================
# Dataset
# ============================================================

class MultimodalColorDataset(Dataset):
    """Dataset that returns tokenized text + processed image + label."""

    def __init__(
        self,
        dataframe,
        encoded_labels,
        img_dir,
        tokenizer,
        image_processor,
        max_len,
        valid_indices=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.encoded_labels = encoded_labels
        self.img_dir = str(img_dir).rstrip("/")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.valid_indices = valid_indices or list(range(len(self.df)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        text = f"{row['item_name']} {row['item_caption']}"
        img_path = os.path.join(self.img_dir, row["image_file_name"])

        try:
            image_arr = load_image_as_rgb_array(img_path)
        except Exception:
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        text_enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )
        img_enc = self.image_processor(
            images=image_arr,
            return_tensors="pt",
            input_data_format="channels_last",
        )

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "label": torch.tensor(self.encoded_labels[real_idx], dtype=torch.float32),
        }


class InferenceDataset(Dataset):
    """Dataset for inference — no labels, returns row_index."""

    def __init__(
        self,
        dataframe,
        img_dir,
        tokenizer,
        image_processor,
        max_len,
        valid_indices,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = str(img_dir).rstrip("/")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        text = f"{row['item_name']} {row['item_caption']}"
        img_path = os.path.join(self.img_dir, row["image_file_name"])

        try:
            image_arr = load_image_as_rgb_array(img_path)
        except Exception:
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        text_enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )
        img_enc = self.image_processor(
            images=image_arr,
            return_tensors="pt",
            input_data_format="channels_last",
        )

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "row_index": real_idx,
        }


# ============================================================
# Data / Eval Helpers
# ============================================================

def build_valid_indices(df, img_dir):
    """Check which images exist and are loadable."""
    valid, skipped = [], 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        path = os.path.join(str(img_dir), row["image_file_name"])
        try:
            with Image.open(path) as im:
                im.convert("RGB")
                w, h = im.size
            if w < 2 or h < 2:
                skipped += 1
                continue
            valid.append(i)
        except Exception:
            skipped += 1
    print(f"  {len(valid)} valid, {skipped} skipped")
    return valid


def prepare_split_data(split="train", val_ratio=0.1, mlb_path=None):
    """Load DB data, parse labels, split, fit binarizer."""
    df_x, df_y = get_split_data(split=split)

    df_y["color_tags"] = df_y["color_tags"].apply(lambda x: x.split(",") if x else [])

    x_train, x_val, y_tags_train, y_tags_val = train_test_split(
        df_x,
        df_y["color_tags"],
        test_size=val_ratio,
        random_state=42,
        shuffle=True,
    )

    mlb = MultiLabelBinarizer()
    y_train_vec = mlb.fit_transform(y_tags_train)
    y_val_vec = mlb.transform(y_tags_val)

    final_mlb_path = mlb_path or ICE_CONFIG["mlb_path"]
    with open(final_mlb_path, "wb") as f:
        pickle.dump(mlb, f)

    print(f"  Label classes ({len(mlb.classes_)}): {list(mlb.classes_)}")
    return x_train, x_val, y_train_vec, y_val_vec, mlb


def ensure_min_predictions(probs, threshold, min_preds=1):
    """Apply threshold, but guarantee at least min_preds per row."""
    preds = (probs > threshold).int().cpu().numpy()
    for i in range(len(preds)):
        if preds[i].sum() < min_preds:
            top_idx = torch.topk(probs[i], min_preds).indices.cpu().numpy()
            preds[i][top_idx] = 1
    return preds


def evaluate_model(model, dataloader, loss_fn, device, threshold=0.5):
    model.eval()
    use_amp = device == "cuda"
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            px = batch["pixel_values"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask, px)
                loss = loss_fn(logits, target)

            preds = (torch.sigmoid(logits.float()) > threshold).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(target.cpu().numpy())
            losses.append(loss.item())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def generate_predictions(
    model,
    df_x,
    img_dir,
    tokenizer,
    image_processor,
    mlb,
    device,
    batch_size,
    max_len,
    threshold=0.3,
    min_preds=1,
    out_path=None,
):
    """Run inference on a dataframe and save predictions CSV."""
    model.eval()
    df_reset = df_x.reset_index(drop=True)
    valid_idx = build_valid_indices(df_reset, img_dir)

    infer_ds = InferenceDataset(
        df_reset,
        img_dir,
        tokenizer,
        image_processor,
        max_len=max_len,
        valid_indices=valid_idx,
    )

    num_workers = min(4, os.cpu_count() or 1)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    use_amp = device == "cuda"
    all_row_indices, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting"):
            px = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask, px)
                probs = torch.sigmoid(logits)

            preds = ensure_min_predictions(probs.float(), threshold, min_preds)
            all_preds.append(preds)
            all_row_indices.extend(batch["row_index"].tolist())

    decoded = mlb.inverse_transform(np.vstack(all_preds))
    results = pd.DataFrame({
        "image_file_name": df_reset.loc[all_row_indices, "image_file_name"].values,
        "color_tags": [list(tags) for tags in decoded],
    })

    if out_path:
        results.to_csv(out_path, index=False)
        print(f"  Saved {len(results)} predictions to {out_path}")
    return results


def load_champion_model_if_exists(client: MlflowClient, model_name: str, alias: str, device: str):
    mv = get_version_by_alias(client, model_name, alias)
    if mv is None:
        return None, None

    model_uri = f"models:/{model_name}@{alias}"
    print(f"  Loading current champion from {model_uri}")
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.to(device)
    model.eval()
    return model, mv


# ============================================================
# Training
# ============================================================

def train(config=None):
    """
    Full ICE training loop with:
    - MLflow experiment tracking
    - epoch-level metrics
    - artifact logging
    - MLflow Model Registry
    - previous champion comparison
    - champion/candidate aliasing
    """
    cfg = {**ICE_CONFIG, **(config or {})}
    cfg_clean = sanitize_params(cfg)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    ensure_registered_model(client, MLFLOW_REGISTERED_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model_id"])
    image_processor = CLIPImageProcessor.from_pretrained(cfg["vision_model_id"])

    X_tr, X_va, y_tr, y_va, mlb = prepare_split_data(
        split=cfg["db_train"],
        val_ratio=float(cfg["val_ratio"]),
        mlb_path=cfg["mlb_path"],
    )
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}")

    img_dir = cfg["image_dir"]
    print("Validating training images...")
    train_valid = build_valid_indices(X_tr.reset_index(drop=True), img_dir)
    print("Validating validation images...")
    val_valid = build_valid_indices(X_va.reset_index(drop=True), img_dir)

    train_ds = MultimodalColorDataset(
        X_tr,
        y_tr,
        img_dir,
        tokenizer,
        image_processor,
        max_len=cfg["max_len"],
        valid_indices=train_valid,
    )
    val_ds = MultimodalColorDataset(
        X_va,
        y_va,
        img_dir,
        tokenizer,
        image_processor,
        max_len=cfg["max_len"],
        valid_indices=val_valid,
    )

    num_workers = min(4, os.cpu_count() or 1)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    dual_encoder = DualEncoder(cfg["text_model_id"], cfg["vision_model_id"]).to(device)
    dual_encoder.unfreeze_encoder_layers(num_layers=cfg["unfreeze_layers"])

    num_classes = len(mlb.classes_)
    classifier = ColorClassifier(input_dim=1536, num_colors=num_classes).to(device)
    model = ICEModel(dual_encoder, classifier).to(device)

    label_counts = y_tr.sum(axis=0).astype(float)
    neg_counts = len(y_tr) - label_counts
    pos_weight_values = np.clip(neg_counts / (label_counts + 1e-6), 1.0, 10.0)

    pos_weights = torch.tensor(pos_weight_values, dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    encoder_params = [p for p in dual_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": classifier.parameters(), "lr": cfg["learning_rate"]},
        {"params": encoder_params, "lr": cfg["encoder_lr"]},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    early_stopping = EarlyStopping(patience=cfg["es_patience"])

    data_manifest = {
        "train_rows": int(len(X_tr)),
        "val_rows": int(len(X_va)),
        "train_valid_images": int(len(train_valid)),
        "val_valid_images": int(len(val_valid)),
        "num_classes": int(len(mlb.classes_)),
        "label_classes": list(map(str, mlb.classes_)),
        "x_train_sha256": dataframe_sha256(X_tr),
        "x_val_sha256": dataframe_sha256(X_va),
        "y_train_sha256": json_sha256(y_tr.tolist()),
        "y_val_sha256": json_sha256(y_va.tolist()),
        "config_sha256": json_sha256(cfg_clean),
        "image_dir": str(img_dir),
        "db_train_split": str(cfg["db_train"]),
        "val_ratio": float(cfg["val_ratio"]),
    }
    data_manifest["dataset_version"] = json_sha256(data_manifest)

    champion_model, champion_version = load_champion_model_if_exists(
        client=client,
        model_name=MLFLOW_REGISTERED_MODEL_NAME,
        alias=MLFLOW_CHAMPION_ALIAS,
        device=device,
    )

    with mlflow.start_run(run_name="ice_dual_encoder_train") as run:
        run_id = run.info.run_id

        mlflow.log_params(cfg_clean)
        mlflow.set_tags({
            "model_family": "ICE DualEncoder",
            "framework": "PyTorch",
            "task": "multilabel-color-classification",
            "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
            "dataset_version": data_manifest["dataset_version"],
            "device": device,
            "amp_enabled": str(use_amp),
            "previous_champion_exists": str(champion_version is not None),
        })

        log_json_artifact("data_manifest.json", data_manifest, artifact_path="metadata")
        log_json_artifact(
            "training_environment.json",
            {
                "device": device,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "num_workers": num_workers,
                "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
                "champion_alias": MLFLOW_CHAMPION_ALIAS,
                "candidate_alias": MLFLOW_CANDIDATE_ALIAS,
            },
            artifact_path="metadata",
        )
        log_json_artifact(
            "class_balance.json",
            {
                "label_counts": label_counts.tolist(),
                "neg_counts": neg_counts.tolist(),
                "pos_weight_min": float(pos_weight_values.min()),
                "pos_weight_max": float(pos_weight_values.max()),
                "pos_weight_mean": float(pos_weight_values.mean()),
            },
            artifact_path="metadata",
        )

        print(
            f"\nTraining ICE "
            f"(max_epochs={cfg['max_epochs']}, batch={cfg['batch_size']}, amp={'BF16' if use_amp else 'OFF'})\n"
        )

        epochs_completed = 0

        for epoch in range(cfg["max_epochs"]):
            model.train()
            train_preds, train_labels = [], []
            train_losses = []

            for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg['max_epochs']} [Train]"):
                optimizer.zero_grad(set_to_none=True)

                px = batch["pixel_values"].to(device, non_blocking=True)
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                target = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(ids, mask, px)
                    loss = loss_fn(logits, target)

                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits.float()) > cfg["train_threshold"]).int().cpu().numpy()
                train_preds.append(preds)
                train_labels.append(target.cpu().numpy())
                train_losses.append(loss.item())

            train_y_true = np.vstack(train_labels)
            train_y_pred = np.vstack(train_preds)

            train_loss = float(np.mean(train_losses))
            train_micro_f1 = float(f1_score(train_y_true, train_y_pred, average="micro", zero_division=0))
            train_macro_f1 = float(f1_score(train_y_true, train_y_pred, average="macro", zero_division=0))

            val_metrics = evaluate_model(
                model=model,
                dataloader=val_dl,
                loss_fn=loss_fn,
                device=device,
                threshold=cfg["val_threshold"],
            )

            current_lrs = [group["lr"] for group in optimizer.param_groups]

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_f1_micro": train_micro_f1,
                "train_f1_macro": train_macro_f1,
                "val_loss": val_metrics["loss"],
                "val_f1_micro": val_metrics["micro_f1"],
                "val_f1_macro": val_metrics["macro_f1"],
                "lr_classifier": float(current_lrs[0]),
                "lr_encoder": float(current_lrs[1]) if len(current_lrs) > 1 else float(current_lrs[0]),
            }, step=epoch + 1)

            print(
                f"\n  Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f} "
                f"train_micro_f1={train_micro_f1:.4f} "
                f"train_macro_f1={train_macro_f1:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_micro_f1={val_metrics['micro_f1']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            scheduler.step(val_metrics["loss"])
            epochs_completed = epoch + 1

            ckpt_path = MODEL_DIR / f"color_model_ep{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)

            if early_stopping(val_metrics["micro_f1"], model, epoch_idx=epoch + 1):
                break

        early_stopping.restore_best_weights(model)

        final_metrics = evaluate_model(
            model=model,
            dataloader=val_dl,
            loss_fn=loss_fn,
            device=device,
            threshold=cfg["val_threshold"],
        )

        best_path = cfg["checkpoint_path"]
        torch.save(model.state_dict(), best_path)
        print(f"\n  Best model saved to {best_path}")

        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(cfg["mlb_path"]), artifact_path="artifacts")

        pred_path = cfg["predictions_path"]
        generate_predictions(
            model=model,
            df_x=X_va,
            img_dir=img_dir,
            tokenizer=tokenizer,
            image_processor=image_processor,
            mlb=mlb,
            device=device,
            batch_size=cfg["batch_size"],
            max_len=cfg["max_len"],
            threshold=cfg["val_threshold"],
            out_path=pred_path,
        )
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        mlflow.log_metrics({
            "best_val_loss": final_metrics["loss"],
            "best_val_f1_micro": final_metrics["micro_f1"],
            "best_val_f1_macro": final_metrics["macro_f1"],
            "epochs_completed": epochs_completed,
            "early_stopping_best_score": float(early_stopping.best_score),
            "early_stopping_best_epoch": int(early_stopping.best_epoch),
            "train_valid_images_count": int(len(train_valid)),
            "val_valid_images_count": int(len(val_valid)),
            "num_classes": int(len(mlb.classes_)),
        })

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            pip_requirements=[
                "torch",
                "transformers",
                "scikit-learn",
                "pillow",
                "numpy",
                "pandas",
                "mlflow",
                "fugashi",
                "unidic-lite",
            ],
        )

        model_version = get_model_version_for_run(
            client=client,
            model_name=MLFLOW_REGISTERED_MODEL_NAME,
            run_id=run_id,
        )
        if model_version is None:
            raise RuntimeError("Could not resolve registered model version for current MLflow run.")

        current_version = str(model_version.version)

        champion_metrics = None
        champion_score = None

        if champion_model is not None:
            champion_metrics = evaluate_model(
                model=champion_model,
                dataloader=val_dl,
                loss_fn=loss_fn,
                device=device,
                threshold=cfg["val_threshold"],
            )
            champion_score = champion_metrics["micro_f1"]

            mlflow.log_metrics({
                "previous_champion_val_loss": champion_metrics["loss"],
                "previous_champion_val_f1_micro": champion_metrics["micro_f1"],
                "previous_champion_val_f1_macro": champion_metrics["macro_f1"],
            })

        challenger_score = final_metrics["micro_f1"]
        promote_new_model = is_better_model(challenger_score, champion_score)

        set_registered_model_alias(
            client=client,
            model_name=MLFLOW_REGISTERED_MODEL_NAME,
            alias=MLFLOW_CANDIDATE_ALIAS,
            version=current_version,
        )

        set_model_version_tags(
            client=client,
            model_name=MLFLOW_REGISTERED_MODEL_NAME,
            version=current_version,
            tags={
                "run_id": run_id,
                "dataset_version": data_manifest["dataset_version"],
                "val_f1_micro": challenger_score,
                "val_f1_macro": final_metrics["macro_f1"],
                "status": "champion" if promote_new_model else "candidate",
            },
        )

        if promote_new_model:
            set_registered_model_alias(
                client=client,
                model_name=MLFLOW_REGISTERED_MODEL_NAME,
                alias=MLFLOW_CHAMPION_ALIAS,
                version=current_version,
            )
            mlflow.set_tag("model_selection_result", "new_model_promoted_to_champion")
        else:
            mlflow.set_tag("model_selection_result", "existing_champion_kept")

        mlflow.log_metrics({
            "new_model_score_for_selection": challenger_score,
            "best_model_score_after_selection": challenger_score if promote_new_model else champion_score,
        })

        try:
            save_run(run_id, "ice_dual_encoder", challenger_score, cfg_clean)
        except Exception as exc:
            print(f"  Warning: save_run failed: {exc}")

        print("\nTraining finished.")
        print(f"  Run ID: {run_id}")
        print(f"  Model URI: {model_info.model_uri}")
        print(f"  Registered model: {MLFLOW_REGISTERED_MODEL_NAME}")
        print(f"  Current version: {current_version}")
        print(f"  Selection result: {'NEW CHAMPION' if promote_new_model else 'OLD CHAMPION KEPT'}")

        return {
            "run_id": run_id,
            "model_uri": model_info.model_uri,
            "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
            "model_version": current_version,
            "best_val_metrics": final_metrics,
            "previous_champion_metrics": champion_metrics,
            "promote_new_model": promote_new_model,
            "dataset_version": data_manifest["dataset_version"],
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICE DualEncoder with MLflow Registry")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--train_threshold", type=float, default=None)
    parser.add_argument("--val_threshold", type=float, default=None)
    parser.add_argument("--unfreeze_layers", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.encoder_lr is not None:
        overrides["encoder_lr"] = args.encoder_lr
    if args.val_ratio is not None:
        overrides["val_ratio"] = args.val_ratio
    if args.train_threshold is not None:
        overrides["train_threshold"] = args.train_threshold
    if args.val_threshold is not None:
        overrides["val_threshold"] = args.val_threshold
    if args.unfreeze_layers is not None:
        overrides["unfreeze_layers"] = args.unfreeze_layers

    train(config=overrides)