"""
ICE DualEncoder — SQL-first training with MLflow tracking, reports,
registry integration, champion/candidate comparison, and DB run logging.

Usage:
    python -m src.models.train_model_ice_sql_mlflow
    python -m src.models.train_model_ice_sql_mlflow --epochs 15 --batch_size 8
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, kein GUI nötig
import matplotlib.pyplot as plt

import os
import sys
import json
import copy
import pickle
import hashlib
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

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
# Helpers
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


def log_json_artifact(filename: str, payload: dict, artifact_path: str = "metadata") -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / filename
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(out), artifact_path=artifact_path)


def save_line_plot(x, y_series, title, y_label, out_path):
    plt.figure(figsize=(8, 5))
    for label, values in y_series.items():
        plt.plot(x, values, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_bar_plot(labels, values, title, y_label, out_path, rotation=90):
    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def multilabel_metrics(y_true, y_pred):
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_label_metrics_df(y_true, y_pred, class_names):
    rows = []
    for i, cls in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "label": str(cls),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
        })
    return pd.DataFrame(rows).sort_values("f1")


# ============================================================
# MLflow Registry helpers
# ============================================================

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


def set_registered_model_alias(client, model_name, alias, version):
    client.set_registered_model_alias(model_name, alias, version)


def set_model_version_tags(client, model_name, version, tags: dict):
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, str(value))


def is_better_model(new_score: float, current_score: float | None) -> bool:
    if current_score is None:
        return True
    return new_score > current_score


# ============================================================
# Model
# ============================================================

class DualEncoder(nn.Module):
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
        return out.last_hidden_state[:, 0, :]

    def get_image_features(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values)
        return out.pooler_output

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
        print(f"  Unfroze last {num_layers} layers: {unfrozen:,} / {total:,} params ({100 * unfrozen / total:.1f}%)")


class ColorClassifier(nn.Module):
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
    def __init__(self, dual_encoder: DualEncoder, classifier: ColorClassifier):
        super().__init__()
        self.dual_encoder = dual_encoder
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, pixel_values):
        img_e = self.dual_encoder.get_image_features(pixel_values)
        txt_e = self.dual_encoder.get_text_features(input_ids, attention_mask)
        return self.classifier(img_e, txt_e)


# ============================================================
# Early stopping
# ============================================================

class EarlyStopping:
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
            print(f"  [ES] No improvement {self.counter}/{self.patience} (best: {self.best_score:.6f})")
            if self.counter >= self.patience:
                print("  [ES] Patience exhausted — stopping.")
                return True
        return False

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"  [ES] Best weights restored (F1={self.best_score:.6f}, epoch={self.best_epoch})")


# ============================================================
# Dataset
# ============================================================

def load_image_as_rgb_array(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[:, :, :3]
    return arr


def build_valid_indices(df, img_dir):
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


class MultimodalColorDataset(Dataset):
    def __init__(self, dataframe, encoded_labels, img_dir, tokenizer, image_processor, max_len, valid_indices=None):
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
            "product_id": int(row["product_id"]),
            "image_file_name": row["image_file_name"],
            "text": text,
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "label": torch.tensor(self.encoded_labels[real_idx], dtype=torch.float32),
        }


# ============================================================
# Data helpers
# ============================================================

def _validate_aligned_split(df_x: pd.DataFrame, df_y: pd.DataFrame, split_name: str):
    if len(df_x) != len(df_y):
        raise ValueError(f"Mismatch in split={split_name}: features={len(df_x)} labels={len(df_y)}")
    if "product_id" not in df_x.columns or "product_id" not in df_y.columns:
        raise ValueError(f"Both df_x and df_y must include product_id for split={split_name}.")
    if not df_x["product_id"].equals(df_y["product_id"]):
        raise ValueError(f"product_id order mismatch for split={split_name}. Fix SQL ordering.")


def prepare_train_val_data(train_split="train", val_split="val", mlb_path=None):
    df_x_train, df_y_train = get_split_data(split=train_split)
    df_x_val, df_y_val = get_split_data(split=val_split)

    _validate_aligned_split(df_x_train, df_y_train, train_split)
    _validate_aligned_split(df_x_val, df_y_val, val_split)

    train_df = df_x_train.copy()
    train_df["color_tags"] = df_y_train["color_tags"].fillna("").apply(lambda x: x.split(",") if x else [])

    val_df = df_x_val.copy()
    val_df["color_tags"] = df_y_val["color_tags"].fillna("").apply(lambda x: x.split(",") if x else [])

    mlb = MultiLabelBinarizer()
    y_train_vec = mlb.fit_transform(train_df["color_tags"])
    y_val_vec = mlb.transform(val_df["color_tags"])

    final_mlb_path = mlb_path or ICE_CONFIG["mlb_path"]
    Path(final_mlb_path).parent.mkdir(parents=True, exist_ok=True)
    with open(final_mlb_path, "wb") as f:
        pickle.dump(mlb, f)

    x_train = train_df.drop(columns=["color_tags"]).reset_index(drop=True)
    x_val = val_df.drop(columns=["color_tags"]).reset_index(drop=True)

    print(f"  Train rows: {len(x_train):,} | Val rows: {len(x_val):,}")
    print(f"  Label classes ({len(mlb.classes_)}): {list(mlb.classes_)}")
    return x_train, x_val, y_train_vec, y_val_vec, mlb


def ensure_min_predictions(probs, threshold, min_preds=1):
    preds = (probs > threshold).int().cpu().numpy()
    for i in range(len(preds)):
        if preds[i].sum() < min_preds:
            top_idx = torch.topk(probs[i], min_preds).indices.cpu().numpy()
            preds[i][top_idx] = 1
    return preds


def evaluate_model(model, dataloader, loss_fn, device, threshold=0.5, min_preds=1):
    model.eval()
    use_amp = device == "cuda"
    losses = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_meta = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            px = batch["pixel_values"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask, px)
                loss = loss_fn(logits, target)

            probs = torch.sigmoid(logits.float())
            preds = ensure_min_predictions(probs, threshold, min_preds=min_preds)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds)
            all_labels.append(target.cpu().numpy())
            losses.append(loss.item())

            for i in range(len(batch["image_file_name"])):
                all_meta.append({
                    "product_id": int(batch["product_id"][i]),
                    "image_file_name": batch["image_file_name"][i],
                    "text": batch["text"][i],
                })

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    metrics = multilabel_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, y_true, y_pred, y_prob, all_meta


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
    cfg = {**ICE_CONFIG, **(config or {})}
    cfg_clean = sanitize_params(cfg)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Configured MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
    print("Configured MLFLOW_EXPERIMENT:", MLFLOW_EXPERIMENT)
    print("Active MLflow tracking URI:", mlflow.get_tracking_uri())

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    ensure_registered_model(client, MLFLOW_REGISTERED_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model_id"])
    image_processor = CLIPImageProcessor.from_pretrained(cfg["vision_model_id"])

    X_tr, X_va, y_tr, y_va, mlb = prepare_train_val_data(
        train_split=cfg.get("db_train", "train"),
        val_split=cfg.get("db_val", "val"),
        mlb_path=cfg["mlb_path"],
    )

    img_dir = cfg["image_dir"]
    print("Validating training images...")
    train_valid = build_valid_indices(X_tr, img_dir)
    print("Validating validation images...")
    val_valid = build_valid_indices(X_va, img_dir)

    train_ds = MultimodalColorDataset(
        X_tr, y_tr, img_dir, tokenizer, image_processor,
        max_len=cfg["max_len"], valid_indices=train_valid
    )
    val_ds = MultimodalColorDataset(
        X_va, y_va, img_dir, tokenizer, image_processor,
        max_len=cfg["max_len"], valid_indices=val_valid
    )

    num_workers = 1
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
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        patience=int(cfg["es_patience"]),
        min_delta=float(cfg.get("es_min_delta", 1e-4)),
    )

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
        "db_train_split": str(cfg.get("db_train", "train")),
        "db_val_split": str(cfg.get("db_val", "val")),
    }
    data_manifest["dataset_version"] = json_sha256(data_manifest)

    champion_model, champion_version = load_champion_model_if_exists(
        client=client,
        model_name=MLFLOW_REGISTERED_MODEL_NAME,
        alias=MLFLOW_CHAMPION_ALIAS,
        device=device,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_f1_micro": [],
        "train_f1_macro": [],
        "train_precision_micro": [],
        "train_precision_macro": [],
        "train_recall_micro": [],
        "train_recall_macro": [],
        "val_loss": [],
        "val_f1_micro": [],
        "val_f1_macro": [],
        "val_precision_micro": [],
        "val_precision_macro": [],
        "val_recall_micro": [],
        "val_recall_macro": [],
        "lr_classifier": [],
        "lr_encoder": [],
        "epoch_seconds": [],
    }

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with mlflow.start_run(run_name="ice_dual_encoder_sql_train") as run:
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
            "source": "sql_first_pipeline",
            "previous_champion_exists": str(champion_version is not None),
        })

        log_json_artifact("data_manifest.json", data_manifest, artifact_path="metadata")
        log_json_artifact("training_environment.json", {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "num_workers": num_workers,
            "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
            "champion_alias": MLFLOW_CHAMPION_ALIAS,
            "candidate_alias": MLFLOW_CANDIDATE_ALIAS,
        }, artifact_path="metadata")
        log_json_artifact("class_balance.json", {
            "label_counts": label_counts.tolist(),
            "neg_counts": neg_counts.tolist(),
            "pos_weight_min": float(pos_weight_values.min()),
            "pos_weight_max": float(pos_weight_values.max()),
            "pos_weight_mean": float(pos_weight_values.mean()),
        }, artifact_path="metadata")

        class_dist_plot = MODEL_DIR / "class_distribution.png"
        save_bar_plot(
            [str(c) for c in mlb.classes_],
            label_counts.tolist(),
            "Class Distribution (Train)",
            "Positive Count",
            class_dist_plot,
            rotation=90,
        )
        mlflow.log_artifact(str(class_dist_plot), artifact_path="reports")

        print(f"\nTraining ICE (max_epochs={cfg['max_epochs']}, batch={cfg['batch_size']}, amp={'BF16' if use_amp else 'OFF'})\n")

        epochs_completed = 0

        for epoch in range(cfg["max_epochs"]):
            epoch_wall_start = time.time()

            if device == "cuda":
                epoch_start = torch.cuda.Event(enable_timing=True)
                epoch_end = torch.cuda.Event(enable_timing=True)
                epoch_start.record()

            model.train()
            train_preds, train_labels, train_losses = [], [], []

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

                probs = torch.sigmoid(logits.float())
                preds = ensure_min_predictions(probs, cfg["train_threshold"], min_preds=1)

                train_preds.append(preds)
                train_labels.append(target.cpu().numpy())
                train_losses.append(loss.item())

            train_y_true = np.vstack(train_labels)
            train_y_pred = np.vstack(train_preds)
            train_metrics = multilabel_metrics(train_y_true, train_y_pred)
            train_loss = float(np.mean(train_losses))

            val_metrics, final_y_true, final_y_pred, final_y_prob, final_meta = evaluate_model(
                model=model,
                dataloader=val_dl,
                loss_fn=loss_fn,
                device=device,
                threshold=cfg["val_threshold"],
                min_preds=1,
            )

            current_lrs = [group["lr"] for group in optimizer.param_groups]

            if device == "cuda":
                epoch_end.record()
                torch.cuda.synchronize()
                epoch_seconds = epoch_start.elapsed_time(epoch_end) / 1000.0
            else:
                epoch_seconds = time.time() - epoch_wall_start

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_f1_micro": train_metrics["micro_f1"],
                "train_f1_macro": train_metrics["macro_f1"],
                "train_precision_micro": train_metrics["micro_precision"],
                "train_precision_macro": train_metrics["macro_precision"],
                "train_recall_micro": train_metrics["micro_recall"],
                "train_recall_macro": train_metrics["macro_recall"],
                "val_loss": val_metrics["loss"],
                "val_f1_micro": val_metrics["micro_f1"],
                "val_f1_macro": val_metrics["macro_f1"],
                "val_precision_micro": val_metrics["micro_precision"],
                "val_precision_macro": val_metrics["macro_precision"],
                "val_recall_micro": val_metrics["micro_recall"],
                "val_recall_macro": val_metrics["macro_recall"],
                "lr_classifier": float(current_lrs[0]),
                "lr_encoder": float(current_lrs[1]) if len(current_lrs) > 1 else float(current_lrs[0]),
                "epoch_seconds": float(epoch_seconds),
            }, step=epoch + 1)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_f1_micro"].append(train_metrics["micro_f1"])
            history["train_f1_macro"].append(train_metrics["macro_f1"])
            history["train_precision_micro"].append(train_metrics["micro_precision"])
            history["train_precision_macro"].append(train_metrics["macro_precision"])
            history["train_recall_micro"].append(train_metrics["micro_recall"])
            history["train_recall_macro"].append(train_metrics["macro_recall"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1_micro"].append(val_metrics["micro_f1"])
            history["val_f1_macro"].append(val_metrics["macro_f1"])
            history["val_precision_micro"].append(val_metrics["micro_precision"])
            history["val_precision_macro"].append(val_metrics["macro_precision"])
            history["val_recall_micro"].append(val_metrics["micro_recall"])
            history["val_recall_macro"].append(val_metrics["macro_recall"])
            history["lr_classifier"].append(float(current_lrs[0]))
            history["lr_encoder"].append(float(current_lrs[1]) if len(current_lrs) > 1 else float(current_lrs[0]))
            history["epoch_seconds"].append(float(epoch_seconds))

            ckpt_path = MODEL_DIR / f"color_model_ep{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

            scheduler.step(val_metrics["loss"])
            epochs_completed = epoch + 1

            print(
                f"\n  Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f} "
                f"train_micro_f1={train_metrics['micro_f1']:.4f} "
                f"train_macro_f1={train_metrics['macro_f1']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_micro_f1={val_metrics['micro_f1']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if early_stopping(val_metrics["micro_f1"], model, epoch_idx=epoch + 1):
                break

        early_stopping.restore_best_weights(model)

        final_metrics, final_y_true, final_y_pred, final_y_prob, final_meta = evaluate_model(
            model=model,
            dataloader=val_dl,
            loss_fn=loss_fn,
            device=device,
            threshold=cfg["val_threshold"],
            min_preds=1,
        )

        best_path = cfg["checkpoint_path"]
        Path(best_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), best_path)
        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

        mlflow.log_artifact(str(cfg["mlb_path"]), artifact_path="artifacts")

        hist_path = MODEL_DIR / "history.csv"
        pd.DataFrame(history).to_csv(hist_path, index=False)

        loss_plot = MODEL_DIR / "loss_curve.png"
        save_line_plot(
            history["epoch"],
            {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
            "Train vs Validation Loss",
            "Loss",
            loss_plot,
        )

        f1_plot = MODEL_DIR / "f1_micro_curve.png"
        save_line_plot(
            history["epoch"],
            {"train_f1_micro": history["train_f1_micro"], "val_f1_micro": history["val_f1_micro"]},
            "Train vs Validation Micro-F1",
            "Micro-F1",
            f1_plot,
        )

        lr_plot = MODEL_DIR / "lr_curve.png"
        save_line_plot(
            history["epoch"],
            {"lr_classifier": history["lr_classifier"], "lr_encoder": history["lr_encoder"]},
            "Learning Rate Schedule",
            "Learning Rate",
            lr_plot,
        )

        epoch_time_plot = MODEL_DIR / "epoch_time_curve.png"
        save_line_plot(
            history["epoch"],
            {"epoch_seconds": history["epoch_seconds"]},
            "Epoch Runtime",
            "Seconds",
            epoch_time_plot,
        )

        per_label_df = per_label_metrics_df(final_y_true, final_y_pred, mlb.classes_)
        per_label_path = MODEL_DIR / "per_label_metrics.csv"
        per_label_df.to_csv(per_label_path, index=False)

        worst_labels_path = MODEL_DIR / "worst_10_labels.csv"
        per_label_df.head(10).to_csv(worst_labels_path, index=False)

        best_labels_path = MODEL_DIR / "best_10_labels.csv"
        per_label_df.sort_values("f1", ascending=False).head(10).to_csv(best_labels_path, index=False)

        top_sample_rows = []
        for i in range(min(len(final_y_prob), 50)):
            true_labels = [str(mlb.classes_[j]) for j in np.where(final_y_true[i] == 1)[0]]
            pred_labels = [str(mlb.classes_[j]) for j in np.where(final_y_pred[i] == 1)[0]]
            top_indices = np.argsort(final_y_prob[i])[::-1][:5]
            top_probs = {str(mlb.classes_[j]): float(final_y_prob[i][j]) for j in top_indices}
            top_sample_rows.append({
                "product_id": final_meta[i]["product_id"],
                "image_file_name": final_meta[i]["image_file_name"],
                "text": final_meta[i]["text"],
                "true_labels": ", ".join(true_labels),
                "pred_labels": ", ".join(pred_labels),
                "top5_probs": json.dumps(top_probs, ensure_ascii=False),
            })

        sample_preds_path = MODEL_DIR / "sample_predictions.csv"
        pd.DataFrame(top_sample_rows).to_csv(sample_preds_path, index=False)

        final_summary = {
            "run_id": run_id,
            "best_epoch": int(early_stopping.best_epoch),
            "epochs_completed": int(epochs_completed),
            "best_val_f1_micro": float(final_metrics["micro_f1"]),
            "best_val_f1_macro": float(final_metrics["macro_f1"]),
            "best_val_precision_micro": float(final_metrics["micro_precision"]),
            "best_val_precision_macro": float(final_metrics["macro_precision"]),
            "best_val_recall_micro": float(final_metrics["micro_recall"]),
            "best_val_recall_macro": float(final_metrics["macro_recall"]),
            "dataset_version": data_manifest["dataset_version"],
        }
        final_summary_path = MODEL_DIR / "final_summary.json"
        final_summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(hist_path), artifact_path="reports")
        mlflow.log_artifact(str(loss_plot), artifact_path="reports")
        mlflow.log_artifact(str(f1_plot), artifact_path="reports")
        mlflow.log_artifact(str(lr_plot), artifact_path="reports")
        mlflow.log_artifact(str(epoch_time_plot), artifact_path="reports")
        mlflow.log_artifact(str(per_label_path), artifact_path="analysis")
        mlflow.log_artifact(str(worst_labels_path), artifact_path="analysis")
        mlflow.log_artifact(str(best_labels_path), artifact_path="analysis")
        mlflow.log_artifact(str(sample_preds_path), artifact_path="predictions")
        mlflow.log_artifact(str(final_summary_path), artifact_path="metadata")

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
                "matplotlib",
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
            champion_metrics, _, _, _, _ = evaluate_model(
                model=champion_model,
                dataloader=val_dl,
                loss_fn=loss_fn,
                device=device,
                threshold=cfg["val_threshold"],
                min_preds=1,
            )
            champion_score = champion_metrics["micro_f1"]

            mlflow.log_metrics({
                "previous_champion_val_loss": champion_metrics["loss"],
                "previous_champion_val_f1_micro": champion_metrics["micro_f1"],
                "previous_champion_val_f1_macro": champion_metrics["macro_f1"],
                "previous_champion_val_precision_micro": champion_metrics["micro_precision"],
                "previous_champion_val_precision_macro": champion_metrics["macro_precision"],
                "previous_champion_val_recall_micro": champion_metrics["micro_recall"],
                "previous_champion_val_recall_macro": champion_metrics["macro_recall"],
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

        best_score_after_selection = challenger_score if promote_new_model else (
            champion_score if champion_score is not None else challenger_score
        )

        mlflow.log_metrics({
            "best_val_loss": final_metrics["loss"],
            "best_val_f1_micro": final_metrics["micro_f1"],
            "best_val_f1_macro": final_metrics["macro_f1"],
            "best_val_precision_micro": final_metrics["micro_precision"],
            "best_val_precision_macro": final_metrics["macro_precision"],
            "best_val_recall_micro": final_metrics["micro_recall"],
            "best_val_recall_macro": final_metrics["macro_recall"],
            "epochs_completed": epochs_completed,
            "early_stopping_best_score": float(early_stopping.best_score),
            "early_stopping_best_epoch": int(early_stopping.best_epoch),
            "train_valid_images_count": int(len(train_valid)),
            "val_valid_images_count": int(len(val_valid)),
            "num_classes": int(len(mlb.classes_)),
            "new_model_score_for_selection": challenger_score,
            "best_model_score_after_selection": float(best_score_after_selection),
        })

        run_payload = {
            **cfg_clean,
            "best_val_f1_micro": float(final_metrics["micro_f1"]),
            "best_val_f1_macro": float(final_metrics["macro_f1"]),
            "epochs_completed": int(epochs_completed),
            "dataset_version": data_manifest["dataset_version"],
            "model_version": current_version,
            "promoted_to_champion": bool(promote_new_model),
        }

        try:
            save_run(run_id, "ice_dual_encoder", challenger_score, run_payload)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICE DualEncoder with SQL + MLflow Registry")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--train_threshold", type=float, default=None)
    parser.add_argument("--val_threshold", type=float, default=None)
    parser.add_argument("--unfreeze_layers", type=int, default=None)
    parser.add_argument("--es_min_delta", type=float, default=None)
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
    if args.es_min_delta is not None:
        overrides["es_min_delta"] = args.es_min_delta

    train(config=overrides)