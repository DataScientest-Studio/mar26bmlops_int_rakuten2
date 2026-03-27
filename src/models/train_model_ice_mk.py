"""
ICE (Image-Caption Ensemble) — DualEncoder training.

Architecture:
    Text:   cl-tohoku/bert-base-japanese-v3  -> 768d CLS token
    Vision: openai/clip-vit-base-patch16     -> 768d pooler output
    Head:   concat(768,768) -> 1024 -> 512 -> num_colors (MLP)

All hyperparameters come from src.config.ICE_CONFIG.
Trained weights are saved to models/color_model_best.pth.

Usage:
    python -m src.models.train_model_ice
    python -m src.models.train_model_ice --epochs 10 --batch_size 64
"""
import os
import ast
import copy
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, CLIPImageProcessor
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ICE_CONFIG, DATA_DIR, MODEL_DIR
from src.db import get_split_data, get_products

import mlflow
import mlflow.pytorch
from src.config import ICE_CONFIG, DATA_DIR, MODEL_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True


# -- Model Architecture ------------------------------------------------

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
        """Unfreeze the last N layers of both encoders for fine-tuning."""
        for layer in self.text_encoder.encoder.layer[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.text_encoder.pooler.parameters():
            p.requires_grad = True

        for layer in self.vision_encoder.vision_model.encoder.layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.vision_encoder.vision_model.post_layernorm.parameters():
            p.requires_grad = True

        unfrozen = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Unfroze last {num_layers} layers: "
              f"{unfrozen:,} / {total:,} params ({100 * unfrozen / total:.1f}%)")


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
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        return self.fc(torch.cat((img_features, txt_features), dim=-1))


# -- Early Stopping ----------------------------------------------------

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float("-inf")
        self.best_weights = None
        self.best_encoder_weights = None

    def __call__(self, score, model, encoder=None):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            if encoder is not None:
                self.best_encoder_weights = copy.deepcopy(encoder.state_dict())
            print(f"  [ES] Val F1 improved to {score:.6f}")
        else:
            self.counter += 1
            print(f"  [ES] No improvement {self.counter}/{self.patience} "
                  f"(best: {self.best_score:.6f})")
            if self.counter >= self.patience:
                print("  [ES] Patience exhausted — stopping.")
                return True
        return False

    def restore_best_weights(self, model, encoder=None):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if encoder is not None and self.best_encoder_weights is not None:
                encoder.load_state_dict(self.best_encoder_weights)
            print(f"  [ES] Best weights restored (F1={self.best_score:.6f})")


# -- Image Loading -----------------------------------------------------

def load_image_as_rgb_array(path: str) -> np.ndarray:
    """Open image and return guaranteed (H, W, 3) uint8 array."""
    image = Image.open(path).convert("RGB")
    arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[:, :, :3]
    return arr


# -- Dataset -----------------------------------------------------------

class MultimodalColorDataset(Dataset):
    """Dataset that returns tokenized text + processed image + label."""

    def __init__(self, dataframe, encoded_labels, img_dir,
                 tokenizer, image_processor, valid_indices=None):
        self.df = dataframe.reset_index(drop=True)
        self.encoded_labels = encoded_labels
        self.img_dir = str(img_dir).rstrip("/")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
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
            text, return_tensors="pt", padding="max_length",
            max_length=ICE_CONFIG["max_len"], truncation=True,
        )
        img_enc = self.image_processor(
            images=image_arr, return_tensors="pt",
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

    def __init__(self, dataframe, img_dir, tokenizer, image_processor, valid_indices):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = str(img_dir).rstrip("/")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
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
            text, return_tensors="pt", padding="max_length",
            max_length=ICE_CONFIG["max_len"], truncation=True,
        )
        img_enc = self.image_processor(
            images=image_arr, return_tensors="pt",
            input_data_format="channels_last",
        )

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "row_index": real_idx,
        }


# -- Helpers -----------------------------------------------------------

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


def prepare_split_data(split="train", val_ratio=0.1):                                                                            # val RATIO
    """Load CSVs, parse labels, split, fit binarizer."""
    df_x,df_y = get_split_data(split = split)

    df_y["color_tags"] = df_y["color_tags"].apply(lambda x: x.split(",") if x else [])

    x_train, x_val, y_tags_train, y_tags_val = train_test_split(
        df_x, df_y["color_tags"], test_size=val_ratio, random_state=42
    )

    mlb = MultiLabelBinarizer()
    y_train_vec = mlb.fit_transform(y_tags_train)
    y_val_vec = mlb.transform(y_tags_val)

    # Save binarizer for inference
    mlb_path = ICE_CONFIG["mlb_path"]
    with open(mlb_path, "wb") as f:
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


def generate_predictions(classifier, dual_encoder, df_x, img_dir,
                         tokenizer, image_processor, mlb, device,
                         threshold=0.3, min_preds=1, out_path=None):
    """Run inference on a dataframe and save predictions CSV."""
    classifier.eval()
    dual_encoder.eval()
    df_reset = df_x.reset_index(drop=True)
    valid_idx = build_valid_indices(df_reset, img_dir)

    infer_ds = InferenceDataset(df_reset, img_dir, tokenizer, image_processor, valid_idx)
    num_workers = min(6, os.cpu_count() or 1)
    infer_loader = DataLoader(
        infer_ds, batch_size=ICE_CONFIG["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    use_amp = (device == "cuda")
    all_row_indices, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting"):
            px = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                img_e = dual_encoder.get_image_features(px)
                txt_e = dual_encoder.get_text_features(ids, mask)
                probs = torch.sigmoid(classifier(img_e, txt_e))

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


# -- Training ----------------------------------------------------------

def train(config=None):
    """
    Full ICE training loop.

    Args:
        config: override dict merged with ICE_CONFIG
    Returns:
        classifier, dual_encoder, mlb, run_id
    """
    cfg = {**ICE_CONFIG, **(config or {})}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="ice_dual_encoder_train") as run:
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        use_amp = (device == "cuda")
        amp_dtype = torch.bfloat16 if use_amp else torch.float32

        # -- Tokenizer, image processor, encoder --
        tokenizer = AutoTokenizer.from_pretrained(cfg["text_model_id"])
        image_processor = CLIPImageProcessor.from_pretrained(cfg["vision_model_id"])

        dual_encoder = DualEncoder(cfg["text_model_id"], cfg["vision_model_id"]).to(device)
        dual_encoder.unfreeze_encoder_layers(num_layers=cfg["unfreeze_layers"])
        dual_encoder.train()

        # -- Data --
        # x_path = DATA_DIR / "processed" / "X_train_processed.csv"                                       mk: 25.03.2026 exchanged trough SQL
        # y_path = DATA_DIR / "processed" / "y_train_processed.csv"
        X_tr, X_va, y_tr, y_va, mlb = prepare_split_data(ICE_CONFIG["db_train"])
        print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}")

        img_dir = cfg["image_dir"]
        print("Validating training images...")
        train_valid = build_valid_indices(X_tr.reset_index(drop=True), img_dir)
        print("Validating validation images...")
        val_valid = build_valid_indices(X_va.reset_index(drop=True), img_dir)

        train_ds = MultimodalColorDataset(X_tr, y_tr, img_dir, tokenizer, image_processor, train_valid)
        val_ds = MultimodalColorDataset(X_va, y_va, img_dir, tokenizer, image_processor, val_valid)

        num_workers = min(6, os.cpu_count() or 1)
        train_dl = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        )
        val_dl = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        )

        # -- Classifier, loss, optimizer, scheduler --
        num_classes = len(mlb.classes_)
        classifier = ColorClassifier(input_dim=1536, num_colors=num_classes).to(device)

        label_counts = y_tr.sum(axis=0).astype(float)
        neg_counts = len(y_tr) - label_counts
        pos_weights = torch.tensor(
            np.clip(neg_counts / (label_counts + 1e-6), 1.0, 10.0), dtype=torch.float32
        ).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        encoder_params = [p for p in dual_encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": classifier.parameters(), "lr": cfg["learning_rate"]},
            {"params": encoder_params, "lr": cfg["encoder_lr"]},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
        )
        early_stopping = EarlyStopping(patience=cfg["es_patience"])

        # -- Training loop --
        max_epochs = cfg["max_epochs"]
        print(f"\nTraining ICE (max {max_epochs} epochs, batch={cfg['batch_size']}, amp=BF16)\n")

        for epoch in range(max_epochs):
            # Train
            classifier.train()
            dual_encoder.train()
            all_preds, all_labels = [], []
            t_loss = 0.0

            for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{max_epochs} [Train]"):
                optimizer.zero_grad(set_to_none=True)

                px = batch["pixel_values"].to(device, non_blocking=True)
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                target = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    img_e = dual_encoder.get_image_features(px)
                    txt_e = dual_encoder.get_text_features(ids, mask)
                    logits = classifier(img_e, txt_e)
                    loss = loss_fn(logits, target)

                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits.float()) > cfg["train_threshold"]).int().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(target.cpu().numpy())
                t_loss += loss.item()

            train_f1 = f1_score(np.vstack(all_labels), np.vstack(all_preds),
                                average="micro", zero_division=0)

            # Validation
            classifier.eval()
            dual_encoder.eval()
            val_preds, val_labels = [], []
            v_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                    px = batch["pixel_values"].to(device, non_blocking=True)
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    mask = batch["attention_mask"].to(device, non_blocking=True)
                    target = batch["label"].to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        img_e = dual_encoder.get_image_features(px)
                        txt_e = dual_encoder.get_text_features(ids, mask)
                        logits = classifier(img_e, txt_e)
                    v_loss += loss_fn(logits, target).item()

                    preds = (torch.sigmoid(logits.float()) > cfg["val_threshold"]).int().cpu().numpy()
                    val_preds.append(preds)
                    val_labels.append(target.cpu().numpy())

            avg_val_loss = v_loss / len(val_dl)
            val_f1 = f1_score(np.vstack(val_labels), np.vstack(val_preds),
                            average="micro", zero_division=0)
            micro_f1 = f1_score(..., average="micro")
            macro_f1 = f1_score(..., average="macro")

            mlflow.log_metric("f1_micro", micro_f1)
            mlflow.log_metric("f1_macro", macro_f1)
            
            print(f"\n  Epoch {epoch+1}: train_loss={t_loss/len(train_dl):.4f} "
                f"train_f1={train_f1:.4f} | val_loss={avg_val_loss:.4f} val_f1={val_f1:.4f}")

            scheduler.step(avg_val_loss)

            # Save epoch checkpoint
            ckpt_path = MODEL_DIR / f"color_model_ep{epoch+1}.pth"
            torch.save({"classifier": classifier.state_dict(),
                        "dual_encoder": dual_encoder.state_dict()}, ckpt_path)

            if early_stopping(val_f1, classifier, encoder=dual_encoder):
                break

        # -- Save best model --
        early_stopping.restore_best_weights(classifier, encoder=dual_encoder)
        best_path = ICE_CONFIG["checkpoint_path"]
        torch.save({"classifier": classifier.state_dict(),
                    "dual_encoder": dual_encoder.state_dict()}, best_path)
        print(f"\n  Best model saved to {best_path}")

        # -- Save run to DB --
        run_id = run.info.run_id
        try:
            from src.db import save_run
            save_run(run_id, "ice_dual_encoder", early_stopping.best_score, cfg)
        except Exception:
            pass

        # -- Generate validation predictions --
        generate_predictions(
            classifier=classifier, dual_encoder=dual_encoder,
            df_x=X_va, img_dir=img_dir, tokenizer=tokenizer,
            image_processor=image_processor, mlb=mlb, device=device,
            threshold=cfg["val_threshold"],
            out_path=MODEL_DIR / "y_pred_training.csv",
        )

        return classifier, dual_encoder, mlb, run_id


# -- CLI ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICE DualEncoder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["max_epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["learning_rate"] = args.lr

    train(config=overrides)
