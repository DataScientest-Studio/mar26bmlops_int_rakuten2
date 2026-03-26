"""
ICE DualEncoder — Inference script.

Loads trained weights from models/color_model_best.pth and generates
predictions on test data. All model classes are imported from
train_model_ice (no duplication).

Required files:
    models/color_model_best.pth  — checkpoint (classifier + dual_encoder)
    models/mlb.pkl               — fitted MultiLabelBinarizer
    data/X_test_processed.csv    — test features
    data/images/                  — image directory

Usage:
    python -m src.models.predict_model_ice
    python -m src.models.predict_model_ice --threshold 0.3 --batch_size 64
"""
import os
import pickle
import argparse

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ICE_CONFIG, DATA_DIR, MODEL_DIR
from src.db import save_predictions, get_products, get_conn, get_split_data
from src.models.train_model_ice_mk import (
    DualEncoder,
    ColorClassifier,
    InferenceDataset,
    build_valid_indices,
    ensure_min_predictions,
)


def predict(threshold=None, batch_size=None, out_path=None):
    """
    Full inference pipeline:
      1. Load encoder + classifier from checkpoint
      2. Run batched forward passes with autocast
      3. Write y_pred.csv
    """
    threshold = threshold or ICE_CONFIG["val_threshold"]
    batch_size = batch_size or ICE_CONFIG["batch_size"]
    out_path = out_path or (MODEL_DIR / ".." / "reports" / "y_pred.csv")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -- Load tokenizer + image processor --
    from transformers import AutoTokenizer, CLIPImageProcessor
    tokenizer = AutoTokenizer.from_pretrained(ICE_CONFIG["text_model_id"])
    image_processor = CLIPImageProcessor.from_pretrained(ICE_CONFIG["vision_model_id"])

    # -- Load binarizer --
    mlb_path = ICE_CONFIG["mlb_path"]
    print(f"Loading binarizer from {mlb_path}")
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    num_classes = len(mlb.classes_)
    print(f"  {num_classes} classes: {list(mlb.classes_)}")

    # -- Load model weights --
    ckpt_path = ICE_CONFIG["checkpoint_path"]
    print(f"Loading weights from {ckpt_path}")

    dual_encoder = DualEncoder(
        ICE_CONFIG["text_model_id"], ICE_CONFIG["vision_model_id"]
    ).to(device)
    classifier = ColorClassifier(input_dim=1536, num_colors=num_classes).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["classifier"])
    dual_encoder.load_state_dict(checkpoint["dual_encoder"])
    classifier.eval()
    dual_encoder.eval()

    # -- Load test data --
    print(f"Loading test data from db")
    df_test, _ = get_split_data(split="val")                                                       # have to be changed via config
    print(f"  {len(df_test):,} rows")

    img_dir = ICE_CONFIG["image_dir"]
    valid_idx, invalid_idx = _build_valid_invalid(df_test.reset_index(drop=True), img_dir)

    infer_ds = InferenceDataset(df_test, img_dir, tokenizer, image_processor, valid_idx)
    infer_loader = DataLoader(
        infer_ds, batch_size=batch_size, shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device == "cuda"),
    )

    # -- Inference --
    print(f"Running inference (threshold={threshold})...")
    use_amp = (device == "cuda")

    all_scores = []
    all_pred_matrix = []
    predictions = {}

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting"):
            px = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            row_indices = batch["row_index"].tolist()

            with torch.amp.autocast("cuda", enabled=use_amp):
                img_e = dual_encoder.get_image_features(px)
                txt_e = dual_encoder.get_text_features(ids, mask)
                logits = classifier(img_e, txt_e)

            probs = torch.sigmoid(logits)
            preds = ensure_min_predictions(probs, threshold, min_preds=1)
            decoded = mlb.inverse_transform(preds)

            all_scores.append(probs.float().cpu().numpy())
            all_pred_matrix.append(preds)

            for row_idx, tags in zip(row_indices, decoded):
                predictions[row_idx] = list(tags)

        # -- mk db ----------------------------------------------
    score_matrix = np.vstack(all_scores)
    pred_matrix  = np.vstack(all_pred_matrix)

    # product_ids aus DB holen — gleiche Reihenfolge wie df_test
    with get_conn() as conn:
        id_df = pd.read_sql(
            "SELECT id FROM products WHERE split = 'test' ORDER BY id",
            conn
        )
    product_ids = id_df["id"].tolist()

    # invalid rows mit leerem pred auffüllen
    for row_idx in invalid_idx:
        predictions[row_idx] = []
    # DB speichern — nur valide rows
    with get_conn() as conn:
        id_df = pd.read_sql(
            "SELECT id FROM products WHERE split = 'test' ORDER BY id",
            conn
        )

    product_ids = id_df["id"].tolist()
    valid_product_ids = [product_ids[i] for i in sorted(valid_idx)]

    save_predictions(
        product_ids=valid_product_ids,
        color_labels=mlb.classes_,
        score_matrix=score_matrix,
        pred_matrix=pred_matrix,
        run_id=f"ice_pred_{np.random.randint(10000, 99999)}"
    )

    # Fill skipped rows with empty lists
    for row_idx in invalid_idx:
        predictions[row_idx] = []

    # -- Write output --
    output_df = pd.DataFrame({
        "color_tags": [str(predictions[i]) for i in range(len(df_test))]
    })
    output_df.to_csv(out_path, index=True)

    non_empty = sum(1 for t in predictions.values() if t)
    print(f"\n  Saved {len(output_df):,} predictions to {out_path}")
    print(f"  {non_empty:,} with tags ({non_empty / len(df_test):.1%})")
    print(f"  {len(df_test) - non_empty:,} empty")

    return predictions






def _build_valid_invalid(df, img_dir):
    """Wrapper around build_valid_indices that also returns invalid."""
    from PIL import Image
    valid, invalid = [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        path = os.path.join(str(img_dir), row["image_file_name"])
        try:
            with Image.open(path) as im:
                im.convert("RGB")
                w, h = im.size
            if w < 2 or h < 2:
                invalid.append(i)
                continue
            valid.append(i)
        except Exception:
            invalid.append(i)
    print(f"  {len(valid)} valid, {len(invalid)} skipped")
    return valid, invalid


# -- CLI ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICE inference")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--x_test", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    predict(
        threshold=args.threshold,
        batch_size=args.batch_size,
        out_path=args.output,
    )
