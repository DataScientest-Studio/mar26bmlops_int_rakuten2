"""
ICE DualEncoder — Inference script.

Usage:
    python -m src.models.predict_model_ice_mk
    python -m src.models.predict_model_ice_mk --split val
    python -m src.models.predict_model_ice_mk --split test
    python -m src.models.predict_model_ice_mk --threshold 0.3 --batch_size 64
"""

import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ICE_CONFIG, MODEL_DIR
from src.db import save_predictions, get_conn, get_split_data
from src.models.train_model_ice_mk import (
    DualEncoder,
    ColorClassifier,
    ICEModel,
    InferenceDataset,
    ensure_min_predictions,
    build_valid_indices,
)


def predict(threshold=None, batch_size=None, out_path=None, split=None):
    """
    Full inference pipeline:
      1. Load data from DB by split
      2. Load encoder + classifier from checkpoint
      3. Run batched forward passes
      4. Compute F1 if labels available (val / pseudo_test)
      5. Save predictions to CSV and DB
    """
    split = split or ICE_CONFIG.get("predict_split", "val")
    threshold = threshold if threshold is not None else ICE_CONFIG["val_threshold"]
    batch_size = batch_size or ICE_CONFIG["batch_size"]
    out_path = Path(out_path or (MODEL_DIR / ".." / "reports" / "y_pred.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    image_source = ICE_CONFIG.get("image_source", "minio")
    minio_bucket_images = ICE_CONFIG.get("minio_bucket_images")
    minio_image_prefix = ICE_CONFIG.get("minio_image_prefix", "")
    img_dir = ICE_CONFIG["image_dir"]

    print(f"Image source: {image_source}")
    if image_source == "minio":
        print(f"  MinIO bucket: {minio_bucket_images}")
        print(f"  MinIO prefix: {minio_image_prefix}")

    # -- Tokenizer + image processor --
    from transformers import AutoTokenizer, CLIPImageProcessor
    tokenizer = AutoTokenizer.from_pretrained(ICE_CONFIG["text_model_id"])
    image_processor = CLIPImageProcessor.from_pretrained(ICE_CONFIG["vision_model_id"])

    # -- Binarizer --
    print(f"Loading binarizer from {ICE_CONFIG['mlb_path']}")
    with open(ICE_CONFIG["mlb_path"], "rb") as f:
        mlb = pickle.load(f)
    num_classes = len(mlb.classes_)
    print(f"  {num_classes} classes: {list(mlb.classes_)}")

    # -- Model weights --
    print(f"Loading weights from {ICE_CONFIG['checkpoint_path']}")
    dual_encoder = DualEncoder(
        ICE_CONFIG["text_model_id"],
        ICE_CONFIG["vision_model_id"]
    ).to(device)
    classifier = ColorClassifier(
        input_dim=1536,
        num_colors=num_classes
    ).to(device)
    model = ICEModel(dual_encoder, classifier).to(device)

    checkpoint = torch.load(
        ICE_CONFIG["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    # updated training saves full model.state_dict()
    model.load_state_dict(checkpoint)
    model.eval()

    # -- Data from DB --
    print(f"Loading data from DB (split={split})")
    df_test, df_labels = get_split_data(split=split)
    print(f"  {len(df_test):,} rows")

    df_test_reset = df_test.reset_index(drop=True)

    valid_idx = build_valid_indices(
        df_test_reset,
        img_dir=img_dir,
        image_source=image_source,
        minio_bucket_images=minio_bucket_images,
        minio_image_prefix=minio_image_prefix,
    )
    valid_idx_set = set(valid_idx)
    invalid_idx = [i for i in range(len(df_test_reset)) if i not in valid_idx_set]

    infer_ds = InferenceDataset(
        dataframe=df_test_reset,
        img_dir=img_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_len=ICE_CONFIG["max_len"],
        valid_indices=valid_idx,
        image_source=image_source,
        minio_bucket_images=minio_bucket_images,
        minio_image_prefix=minio_image_prefix,
    )

    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device == "cuda"),
        persistent_workers=(min(4, os.cpu_count() or 1) > 0),
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

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask, px)

            probs = torch.sigmoid(logits)
            preds = ensure_min_predictions(probs.float(), threshold, min_preds=1)
            decoded = mlb.inverse_transform(preds)

            all_scores.append(probs.float().cpu().numpy())
            all_pred_matrix.append(preds)
            for row_idx, tags in zip(row_indices, decoded):
                predictions[row_idx] = list(tags)

    if len(all_scores) > 0:
        score_matrix = np.vstack(all_scores)
        pred_matrix = np.vstack(all_pred_matrix)
    else:
        score_matrix = np.empty((0, len(mlb.classes_)))
        pred_matrix = np.empty((0, len(mlb.classes_)), dtype=int)

    # -- Fill invalid rows --
    for row_idx in invalid_idx:
        predictions[row_idx] = []

    # -- F1 if labels available --
    if df_labels is not None and len(df_labels) > 0 and len(valid_idx) > 0:
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import MultiLabelBinarizer as MLB

        df_labels = df_labels.copy()
        df_labels["color_tags"] = df_labels["color_tags"].apply(
            lambda x: x.split(",") if x else []
        )

        mlb_eval = MLB(classes=mlb.classes_)
        mlb_eval.fit([mlb.classes_])

        y_true = mlb_eval.transform(df_labels["color_tags"].tolist())

        # valid_idx are already the rows used for inference
        f1 = f1_score(
            y_true[sorted(valid_idx)],
            pred_matrix,
            average="micro",
            zero_division=0,
        )
        print(f"  F1 micro (split={split}): {f1:.4f}")

    # -- Save to DB --
    with get_conn() as conn:
        query = (
            "SELECT id FROM products WHERE split = %s ORDER BY id"
            if "postgres" in str(type(conn)).lower() or conn.__class__.__module__.startswith("psycopg2")
            else "SELECT id FROM products WHERE split = ? ORDER BY id"
        )
        id_df = pd.read_sql(query, conn, params=(split,))

    product_ids = id_df["id"].tolist()
    valid_product_ids = [product_ids[i] for i in sorted(valid_idx)]

    if len(valid_product_ids) > 0:
        save_predictions(
            product_ids=valid_product_ids,
            color_labels=mlb.classes_,
            score_matrix=score_matrix,
            pred_matrix=pred_matrix,
            run_id=f"ice_pred_{np.random.randint(10000, 99999)}",
        )

    # -- CSV output --
    output_df = pd.DataFrame({
        "color_tags": [str(predictions[i]) for i in range(len(df_test_reset))]
    })
    output_df.to_csv(out_path, index=True)

    non_empty = sum(1 for tags in predictions.values() if tags)
    print(f"\n  Saved {len(output_df):,} predictions to {out_path}")
    print(f"  {non_empty:,} with tags ({(non_empty / len(df_test_reset)):.1%})")
    print(f"  {len(df_test_reset) - non_empty:,} empty")

    return {
        "predictions": predictions,
        "output_path": str(out_path),
        "num_rows": len(output_df),
        "num_non_empty": non_empty,
    }


# -- CLI ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICE inference")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="val | pseudo_test | test (default: val)"
    )
    args = parser.parse_args()

    predict(
        threshold=args.threshold,
        batch_size=args.batch_size,
        out_path=args.output,
        split=args.split,
    )