"""
================================================================================
Color Classifier — Inference Script
================================================================================

PURPOSE
-------
Loads the trained model weights and generates y_pred.csv by running inference
on X_test_processed.csv.  This script is completely standalone — it
does not import anything from train_model.py and can be run independently.

REQUIRED FILES
--------------
  models/color_model_best.pth   — checkpoint dict:
                                  {"classifier": ..., "dual_encoder": ...}
  models/mlb.pkl                — fitted MultiLabelBinarizer from training
  data/X_test_processed_shortened.csv  — test features
  data/images/                  — image directory

OUTPUT
------
  reports/y_pred.csv

      ,color_tags
      0,['Transparent']
      1,['Silver']
      2,"['Brown', 'Khaki']"
      3,[]

  The unnamed first column is the integer index matching the input CSV row
  numbers.  Rows whose image file is missing or corrupt are filled with [].

  Edit the CONFIG block below to adjust paths or the prediction threshold.

================================================================================
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModel,
    CLIPImageProcessor,
)

# Allow PIL to load truncated image files rather than raising OSError.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


# ==========================================
# CONFIG  —  edit paths and settings
# ==========================================
CONFIG = {
    # Paths to artefacts saved during training
    "model_weights": REPO_ROOT / "models" / "color_model_best.pth",
    "mlb_path":      REPO_ROOT / "models" / "mlb.pkl",

    # Test data
    "x_test_path":   REPO_ROOT / "data" / "X_test_processed_shortened.csv",
    "img_dir":       REPO_ROOT / "data" / "images",

    # Output
    "output_path":   REPO_ROOT / "reports" / "y_pred.csv",

    # Prediction threshold
    "threshold":     0.5,

    # Minimum number of colour tags to always predict per item
    "min_preds":     1,

    # Batch size for inference
    "batch_size":    32,

    # Model IDs — must match exactly what was used during training
    "text_model_id":   "cl-tohoku/bert-base-japanese-v3",
    "vision_model_id": "openai/clip-vit-base-patch16",
}


# ==========================================
# Model Architecture
# ==========================================

class DualEncoder(nn.Module):
    """
    Japanese BERT text encoder + CLIP ViT image encoder.

    All parameters are initially frozen.  The checkpoint loaded from
    models/color_model_best.pth contains fine-tuned weights for the last
    UNFREEZE_LAYERS layers of each encoder (set during training in
    train_model.py).  Those updated weights are restored when
    dual_encoder.load_state_dict(checkpoint["dual_encoder"]) is called.

    Architecture must be identical to train_model.py so that the saved
    weights load without errors.
    """
    def __init__(self, text_model_id: str, vision_model_id: str):
        super().__init__()
        self.text_encoder   = AutoModel.from_pretrained(text_model_id)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_id)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def get_text_features(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def get_image_features(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values)
        return out.pooler_output


class ColorClassifier(nn.Module):
    """
    Three-layer MLP head: 1536 → 1024 → 512 → num_colors.
    Architecture must be identical to train_model.py.
    """
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

    def forward(self, img_features: torch.Tensor,
                txt_features: torch.Tensor) -> torch.Tensor:
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        return self.fc(torch.cat((img_features, txt_features), dim=-1))


# ==========================================
# Image loading helper
# ==========================================

def load_image_as_rgb_array(path: str) -> np.ndarray:
    """
    Open an image file and return it as a guaranteed (H, W, 3) uint8 numpy
    array suitable for CLIPImageProcessor.

    Args:
        path : absolute path to the image file.

    Returns:
        numpy array of shape (H, W, 3), dtype uint8.
    """
    image = Image.open(path).convert("RGB")
    arr   = np.array(image)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[:, :, :3]

    return arr


# ==========================================
# Inference Dataset
# ==========================================

class InferenceDataset(Dataset):
    """
    Loads items from X_test_processed_shortened.csv for inference.

    Only rows whose image file is valid (confirmed by build_valid_indices)
    are included.  Returns `row_index` (the original CSV row number) so
    that predictions can be aligned back to the correct row in y_pred.csv.
    """

    def __init__(self, dataframe: pd.DataFrame, img_dir: str,
                 tokenizer, image_processor, valid_indices: list):
        self.df              = dataframe.reset_index(drop=True)
        self.img_dir         = str(img_dir).rstrip("/")
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.valid_indices   = valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx      = self.valid_indices[idx]
        row           = self.df.iloc[real_idx]
        text_input    = f"{row['item_name']} {row['item_caption']}"
        full_img_path = os.path.join(self.img_dir, row["image_file_name"])

        try:
            image_arr = load_image_as_rgb_array(full_img_path)
        except Exception:
            # Last-resort fallback: substitute a plain grey 224×224 image so
            # the batch can proceed without crashing
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        text_enc = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        img_enc = self.image_processor(
            images=image_arr,
            return_tensors="pt",
            input_data_format="channels_last",
        )

        return {
            "input_ids":      text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values":   img_enc["pixel_values"].squeeze(0),
            "row_index":      real_idx,
        }


# ==========================================
# Helpers
# ==========================================

def build_valid_indices(df: pd.DataFrame, img_dir: str):
    """
    Return (valid_indices, invalid_indices) for rows in df.

    Does a full pixel decode (not just a header check) to catch truncated
    files that would crash during inference when PIL tries to read missing
    pixel data.  Also filters out 1×1 pixel images which cause
    CLIPImageProcessor to mis-detect the channel axis.

    Args:
        df      : DataFrame containing an "image_file_name" column.
        img_dir : Root directory for image files.

    Returns:
        valid   : list of row indices that are safe to use.
        invalid : list of row indices that were skipped (filled with [] in output).
    """
    valid, invalid = [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        path = os.path.join(str(img_dir), row["image_file_name"])
        try:
            with Image.open(path) as im:
                # Full decode — catches truncated files that pass a header check
                im.convert("RGB")
                w, h = im.size

            # Skip 1×1 pixel images — no visual information, confuses processor
            if w < 2 or h < 2:
                invalid.append(i)
                continue

            valid.append(i)
        except Exception:
            invalid.append(i)

    print(f"  -> {len(valid)} valid, {len(invalid)} skipped "
          f"(will predict [] for skipped rows)")
    return valid, invalid


def ensure_min_predictions(probs: torch.Tensor, threshold: float,
                           min_preds: int = 1) -> np.ndarray:
    """
    Apply threshold as normal, but guarantee at least `min_preds` predictions
    per item so no row ever returns an empty list.

    For any row where no class clears the threshold, the top-N most confident
    classes are forced to 1 regardless of their confidence score.

    Args:
        probs     : sigmoid probabilities, shape (B, num_classes), on CPU.
        threshold : confidence cutoff (e.g. 0.5).
        min_preds : minimum colours to always predict per item (default 1).

    Returns:
        Binary numpy array of shape (B, num_classes).
    """
    preds = (probs > threshold).int().cpu().numpy()

    for i in range(len(preds)):
        if preds[i].sum() < min_preds:
            top_indices           = torch.topk(probs[i], min_preds).indices.cpu().numpy()
            preds[i][top_indices] = 1

    return preds


def format_color_tags(tags: list) -> str:
    """
    Serialise a list of colour strings to match the required CSV format.

    Examples:
        []                 → "[]"
        ['Transparent']    → "['Transparent']"
        ['Brown', 'Khaki'] → "['Brown', 'Khaki']"
    """
    return str(tags)


# ==========================================
# Main inference function
# ==========================================

def predict():
    """
    Full inference pipeline:
      1. Load tokenizer, image processor, and encoders.
      2. Load classifier head and encoder weights from models/.
      3. Build InferenceDataset from data/X_test_processed_shortened.csv.
      4. Run batched forward passes with fp16 autocast (GPU only).
      5. Fill skipped rows with empty lists.
      6. Write reports/y_pred.csv.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Ensure the reports directory exists before writing output
    CONFIG["output_path"].parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load tokenizer, image processor, encoders
    # ------------------------------------------------------------------
    print("\nLoading tokenizer and image processor...")
    tokenizer       = AutoTokenizer.from_pretrained(CONFIG["text_model_id"])
    image_processor = CLIPImageProcessor.from_pretrained(CONFIG["vision_model_id"])

    print("Loading dual encoder...")
    dual_encoder = DualEncoder(
        CONFIG["text_model_id"],
        CONFIG["vision_model_id"]
    ).to(device)

    # ------------------------------------------------------------------
    # Load MultiLabelBinarizer — holds the exact class vocabulary and
    # ordering used during training.  Must come from the same training
    # run as color_model_best.pth.
    # ------------------------------------------------------------------
    print(f"Loading label binarizer from '{CONFIG['mlb_path']}'...")
    with open(CONFIG["mlb_path"], "rb") as f:
        mlb = pickle.load(f)
    num_classes = len(mlb.classes_)
    print(f"  -> {num_classes} colour classes: {list(mlb.classes_)}")

    # ------------------------------------------------------------------
    # Load checkpoint — dict with "classifier" and "dual_encoder" keys.
    # The dual encoder weights include the fine-tuned last N layers from
    # training. These are restored here so inference matches training.
    # map_location handles GPU→CPU or CPU→GPU weight loading automatically.
    # ------------------------------------------------------------------
    print(f"Loading model weights from '{CONFIG['model_weights']}'...")
    checkpoint = torch.load(CONFIG["model_weights"], map_location=device)

    classifier = ColorClassifier(input_dim=1536, num_colors=num_classes).to(device)
    classifier.load_state_dict(checkpoint["classifier"])
    dual_encoder.load_state_dict(checkpoint["dual_encoder"])

    # eval() disables Dropout for deterministic, reproducible predictions
    classifier.eval()
    dual_encoder.eval()
    print("  -> Weights loaded successfully.")

    # ------------------------------------------------------------------
    # Load test data and validate images
    # ------------------------------------------------------------------
    print(f"\nLoading test data from '{CONFIG['x_test_path']}'...")
    df_test = pd.read_csv(CONFIG["x_test_path"])
    print(f"  -> {len(df_test):,} rows loaded.")

    print("Validating test images...")
    valid_idx, invalid_idx = build_valid_indices(
        df_test.reset_index(drop=True), CONFIG["img_dir"]
    )

    # ------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------
    infer_ds = InferenceDataset(
        df_test, CONFIG["img_dir"], tokenizer, image_processor, valid_idx
    )

    infer_loader = DataLoader(
        infer_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,   # must be False — order must match row indices
        num_workers=4,   # use multiple workers for faster data loading
        pin_memory=(device == "cuda"),
    )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    print("\nRunning inference...")

    # autocast runs forward passes in fp16 on GPU Tensor Cores for ~2x
    # throughput.  Disabled automatically on CPU where fp16 is not beneficial.
    use_amp     = (device == "cuda")
    predictions: dict = {}

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting"):
            px          = batch["pixel_values"].to(device)
            ids         = batch["input_ids"].to(device)
            mask        = batch["attention_mask"].to(device)
            row_indices = batch["row_index"].tolist()

            with autocast("cuda", enabled=use_amp):
                img_embeds = dual_encoder.get_image_features(px)
                txt_embeds = dual_encoder.get_text_features(ids, mask)
                logits     = classifier(img_embeds, txt_embeds)

            # Threshold probabilities; ensure at least min_preds per item
            probs = torch.sigmoid(logits)
            preds = ensure_min_predictions(
                probs, CONFIG["threshold"], CONFIG["min_preds"]
            )

            decoded = mlb.inverse_transform(preds)  # list of tuples

            for row_idx, colour_tuple in zip(row_indices, decoded):
                predictions[row_idx] = list(colour_tuple)

    # Fill skipped rows with empty lists so output row count matches input
    for row_idx in invalid_idx:
        predictions[row_idx] = []

    # ------------------------------------------------------------------
    # Build and write output CSV
    # ------------------------------------------------------------------
    color_tags_series = pd.Series(
        [predictions[i] for i in range(len(df_test))],
        name="color_tags"
    )

    output_df = pd.DataFrame(
        {"color_tags": color_tags_series.apply(format_color_tags)}
    )

    # index=True produces the unnamed integer index column:
    #     ,color_tags
    #     0,['Transparent']
    #     1,['Silver']
    output_df.to_csv(CONFIG["output_path"], index=True)
    print(f"\nSaved {len(output_df):,} predictions to '{CONFIG['output_path']}'")

    non_empty = sum(1 for tags in predictions.values() if tags)
    print(f"  -> {non_empty:,} items with at least one colour tag "
          f"({non_empty / len(df_test):.1%})")
    print(f"  -> {len(df_test) - non_empty:,} items with no colour predicted ([])")


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    predict()