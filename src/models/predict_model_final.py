"""
ICE DualEncoder — Inference script using SQL data + MLflow Model Registry.

Default behavior:
    - load inference data from SQL by split
    - load the CHAMPION model from MLflow Model Registry
    - load the matching mlb.pkl artifact from the source training run
    - run batched inference
    - compute F1 for labeled splits (e.g. val / pseudo_test)
    - save predictions to CSV and DB

Usage examples:
    python -m src.models.predict_model_final
    python -m src.models.predict_model_final --split val
    python -m src.models.predict_model_final --split pseudo_test
    python -m src.models.predict_model_final --split test
    python -m src.models.predict_model_final --model-alias champion
    python -m src.models.predict_model_final --model-alias candidate
    python -m src.models.predict_model_final --model-version 3
    python -m src.models.predict_model_final --threshold 0.3 --batch_size 64
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageFile
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    ICE_CONFIG,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_CANDIDATE_ALIAS,
    MINIO_ENDPOINT,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
)
from src.db import save_predictions, get_split_data

# Import model classes so MLflow can deserialize the logged PyTorch model
from src.models.train_model_final import (
    ICEModel,
    load_image_as_rgb_array,
    build_valid_indices,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Helpers
# ============================================================

def configure_mlflow_s3() -> None:
    """
    Ensure MLflow artifact access works when the backend uses MinIO / S3.
    """
    if MINIO_ENDPOINT:
        endpoint = MINIO_ENDPOINT
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = f"http://{endpoint}"

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint
        os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ROOT_USER
        os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_ROOT_PASSWORD
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def ensure_min_predictions(probs: torch.Tensor, threshold: float, min_preds: int = 1) -> np.ndarray:
    preds = (probs > threshold).int().cpu().numpy()
    for i in range(len(preds)):
        if preds[i].sum() < min_preds:
            top_idx = torch.topk(probs[i], min_preds).indices.cpu().numpy()
            preds[i][top_idx] = 1
    return preds


# ============================================================
# Dataset
# ============================================================

class SQLInferenceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        img_dir,
        tokenizer,
        image_processor,
        max_len,
        valid_indices=None,
        image_source="local",
        minio_bucket_images=None,
        minio_image_prefix="",
    ):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = str(img_dir).rstrip("/")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.valid_indices = valid_indices or list(range(len(self.df)))
        self.image_source = image_source
        self.minio_bucket_images = minio_bucket_images
        self.minio_image_prefix = minio_image_prefix

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]

        item_name = "" if pd.isna(row.get("item_name")) else str(row.get("item_name"))
        item_caption = "" if pd.isna(row.get("item_caption")) else str(row.get("item_caption"))
        text = f"{item_name} {item_caption}".strip()

        try:
            image_arr = load_image_as_rgb_array(
                image_file_name=row["image_file_name"],
                image_source=self.image_source,
                img_dir=self.img_dir,
                minio_bucket=self.minio_bucket_images,
                minio_prefix=self.minio_image_prefix,
            )
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
            "row_index": real_idx,
            "product_id": int(row["product_id"]) if "product_id" in row else -1,
            "image_file_name": row["image_file_name"],
            "text": text,
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": img_enc["pixel_values"].squeeze(0),
        }


# ============================================================
# MLflow helpers
# ============================================================

def resolve_model_version(client: MlflowClient, model_name: str, alias: str | None, version: str | None):
    if version is not None:
        mv = client.get_model_version(model_name, str(version))
        return mv, f"models:/{model_name}/{version}"

    alias_to_use = alias or MLFLOW_CHAMPION_ALIAS
    mv = client.get_model_version_by_alias(model_name, alias_to_use)
    return mv, f"models:/{model_name}@{alias_to_use}"


def download_mlb_from_run(client: MlflowClient, run_id: str):
    """
    Training script logs mlb.pkl via:
        mlflow.log_artifact(str(cfg["mlb_path"]), artifact_path="artifacts")
    So we expect the file at artifacts/<filename>.
    """
    artifact_rel_path = f"artifacts/{Path(ICE_CONFIG['mlb_path']).name}"
    local_path = client.download_artifacts(run_id, artifact_rel_path)

    with open(local_path, "rb") as f:
        mlb = pickle.load(f)
    return mlb, local_path


def load_registered_model_and_mlb(
    tracking_uri: str,
    model_name: str,
    model_alias: str | None = None,
    model_version: str | None = None,
    device: str = "cpu",
):
    configure_mlflow_s3()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    mv, model_uri = resolve_model_version(
        client=client,
        model_name=model_name,
        alias=model_alias,
        version=model_version,
    )

    print(f"Loading model from MLflow: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.to(device)
    model.eval()

    source_run_id = mv.run_id
    resolved_version = str(mv.version)
    resolved_alias = model_alias if model_version is None else None

    print(f"Resolved registered model: {model_name}")
    print(f"  source_run_id: {source_run_id}")
    print(f"  model_version: {resolved_version}")

    mlb, mlb_local_path = download_mlb_from_run(client, source_run_id)
    print(f"Loaded binarizer from run artifact: {mlb_local_path}")

    return {
        "model": model,
        "mlb": mlb,
        "model_uri": model_uri,
        "model_name": model_name,
        "model_version": resolved_version,
        "model_alias": resolved_alias,
        "source_run_id": source_run_id,
    }


# ============================================================
# Predict
# ============================================================

def predict(
    threshold=None,
    batch_size=None,
    out_path=None,
    split=None,
    model_alias=None,
    model_version=None,
    model_name=None,
    save_outputs=True,
    save_db=True,
):
    """
    Full inference pipeline:
      1. Load SQL data by split
      2. Resolve champion/candidate/version from MLflow Registry
      3. Load matching mlb.pkl from the source training run
      4. Run batched inference
      5. Compute F1 if labels available
      6. Save predictions to CSV and DB
    """
    split = split or ICE_CONFIG.get("predict_split", "val")
    threshold = threshold if threshold is not None else ICE_CONFIG["val_threshold"]
    batch_size = batch_size or ICE_CONFIG["batch_size"]
    model_name = model_name or MLFLOW_REGISTERED_MODEL_NAME

    out_path = Path(out_path or (MODEL_DIR / ".." / "reports" / f"y_pred_{split}.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load tokenizer / image processor from config
    # --------------------------------------------------------
    from transformers import AutoTokenizer, CLIPImageProcessor

    tokenizer = AutoTokenizer.from_pretrained(ICE_CONFIG["text_model_id"])
    image_processor = CLIPImageProcessor.from_pretrained(ICE_CONFIG["vision_model_id"])

    # --------------------------------------------------------
    # Load model + mlb from MLflow Registry
    # --------------------------------------------------------
    registry_payload = load_registered_model_and_mlb(
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=model_name,
        model_alias=model_alias or MLFLOW_CHAMPION_ALIAS,
        model_version=model_version,
        device=device,
    )

    model = registry_payload["model"]
    mlb = registry_payload["mlb"]
    source_run_id = registry_payload["source_run_id"]
    resolved_model_version = registry_payload["model_version"]
    resolved_model_uri = registry_payload["model_uri"]

    num_classes = len(mlb.classes_)
    print(f"  {num_classes} classes: {list(mlb.classes_)}")

    # --------------------------------------------------------
    # Load data from SQL
    # --------------------------------------------------------
    print(f"Loading data from DB (split={split})")
    df_test, df_labels = get_split_data(split=split)
    print(f"  {len(df_test):,} rows")

    img_dir = ICE_CONFIG["image_dir"]
    image_source = ICE_CONFIG.get("image_source", "local")
    minio_bucket = ICE_CONFIG.get("minio_bucket_images")
    minio_prefix = ICE_CONFIG.get("minio_image_prefix", "")

    print(
        f"Image source: {image_source}"
        + (
            f" (bucket={minio_bucket})"
            if image_source == "minio"
            else f" ({img_dir})"
        )
    )

    valid_idx = build_valid_indices(
        df_test.reset_index(drop=True),
        img_dir,
        image_source,
        minio_bucket,
        minio_prefix,
    )
    valid_idx_set = set(valid_idx)
    invalid_idx = [i for i in range(len(df_test)) if i not in valid_idx_set]

    infer_ds = SQLInferenceDataset(
        dataframe=df_test,
        img_dir=img_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_len=ICE_CONFIG["max_len"],
        valid_indices=valid_idx,
        image_source=image_source,
        minio_bucket_images=minio_bucket,
        minio_image_prefix=minio_prefix,
    )

    num_workers = 4 if device == "cuda" else 0
    prefetch = 2 if device == "cuda" else None

    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=prefetch,
        persistent_workers=(num_workers > 0),
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    print(f"Running inference (threshold={threshold})...")
    use_amp = (device == "cuda")

    all_scores = []
    all_pred_matrix = []
    predictions = {}
    prediction_rows = []

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting"):
            px = batch["pixel_values"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            row_indices = batch["row_index"].tolist()
            product_ids = batch["product_id"].tolist()
            image_names = batch["image_file_name"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask, px)

            probs = torch.sigmoid(logits.float())
            preds = ensure_min_predictions(probs, threshold=threshold, min_preds=1)
            decoded = mlb.inverse_transform(preds)

            probs_np = probs.cpu().numpy()

            all_scores.append(probs_np)
            all_pred_matrix.append(preds)

            for local_i, (row_idx, product_id, image_name, tags) in enumerate(
                zip(row_indices, product_ids, image_names, decoded)
            ):
                tag_list = list(tags)
                predictions[row_idx] = tag_list

                top_indices = np.argsort(probs_np[local_i])[::-1][:5]
                top5_probs = {
                    str(mlb.classes_[j]): float(probs_np[local_i][j])
                    for j in top_indices
                }

                prediction_rows.append({
                    "row_index": int(row_idx),
                    "product_id": int(product_id),
                    "image_file_name": image_name,
                    "pred_labels": ", ".join(tag_list),
                    "top5_probs": json.dumps(top5_probs, ensure_ascii=False),
                    "model_name": model_name,
                    "model_version": resolved_model_version,
                    "source_run_id": source_run_id,
                    "split": split,
                    "threshold": threshold,
                })

    if len(all_scores) == 0:
        raise RuntimeError("No valid images available for inference. Cannot create predictions.")

    score_matrix = np.vstack(all_scores)
    pred_matrix = np.vstack(all_pred_matrix)

    # --------------------------------------------------------
    # Fill invalid rows with empty predictions
    # --------------------------------------------------------
    for row_idx in invalid_idx:
        predictions[row_idx] = []

    # --------------------------------------------------------
    # Compute F1 if labels exist
    # --------------------------------------------------------
    f1_micro = None

    if df_labels is not None and len(df_labels) > 0:
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import MultiLabelBinarizer as MLB

        df_labels = df_labels.copy()
        df_labels["color_tags"] = df_labels["color_tags"].fillna("").apply(
            lambda x: x.split(",") if x else []
        )

        mlb_eval = MLB(classes=mlb.classes_)
        mlb_eval.fit([mlb.classes_])

        y_true = mlb_eval.transform(df_labels["color_tags"].tolist())
        f1_micro = f1_score(
            y_true[sorted(valid_idx)],
            pred_matrix,
            average="micro",
            zero_division=0,
        )
        print(f"  F1 micro (split={split}): {f1_micro:.4f}")

    # --------------------------------------------------------
    # Save predictions to DB
    # --------------------------------------------------------
    db_prediction_run_id = None
    if save_db:
        valid_product_ids = [int(df_test.iloc[i]["product_id"]) for i in sorted(valid_idx)]

        db_prediction_run_id = (
            f"ice_pred_model_{model_name}_v{resolved_model_version}_from_{source_run_id[:8]}"
        )

        print(f"Saving only valid-image predictions to DB: {len(valid_product_ids)} rows")

        save_predictions(
            product_ids=valid_product_ids,
            color_labels=mlb.classes_,
            score_matrix=score_matrix,
            pred_matrix=pred_matrix,
            run_id=db_prediction_run_id,
        )

    # --------------------------------------------------------
    # Save CSV report
    # --------------------------------------------------------
    details_path = None
    submission_path = None
    if save_outputs:
        # Standard output with product_id + image (used by drift + analysis)
        output_df = pd.DataFrame({
            "product_id": df_test["product_id"] if "product_id" in df_test.columns else pd.Series(range(len(df_test))),
            "image_file_name": df_test["image_file_name"],
            "color_tags": [json.dumps(predictions[i], ensure_ascii=False) for i in range(len(df_test))],
        })
        output_df.to_csv(out_path, index=False)

        # Detailed report with top-5 probs (used by drift score-distribution report)
        details_path = out_path.with_name(out_path.stem + "_details.csv")
        pd.DataFrame(prediction_rows).to_csv(details_path, index=False)

        # -- Rakuten challenge submission format (only for test split) --
        # Required:
        #   - unnamed 0-based integer index (NOT product_id)
        #   - color_tags as Python-style list literal: ['Pink', 'Blue']
        #     (pandas default; NOT JSON with escaped double-quotes)
        if split == "test":
            submission_df = pd.DataFrame({
                "color_tags": [predictions[i] for i in range(len(df_test))],
            })
            # index=True with no index name -> leading unnamed column "0,1,2,..."
            submission_path = out_path.with_name("submission.csv")
            submission_df.to_csv(submission_path, index=True, header=True)

        non_empty = sum(1 for t in predictions.values() if t)
        print(f"\nSaved {len(output_df):,} predictions to {out_path}")
        print(f"Saved detailed prediction report to {details_path}")
        if submission_path:
            print(f"Saved Rakuten submission file to {submission_path}")
        print(f"{non_empty:,} with tags ({non_empty / len(df_test):.1%})")
        print(f"{len(df_test) - non_empty:,} empty")

        print("\nModel source:")
        print(f"  model_name:    {model_name}")
        print(f"  model_uri:     {resolved_model_uri}")
        print(f"  model_version: {resolved_model_version}")
        print(f"  source_run_id: {source_run_id}")
        print(f"  db_run_id:     {db_prediction_run_id}")

        return {
            "predictions": predictions,
            "output_csv": str(out_path) if save_outputs else None,
            "details_csv": str(details_path) if details_path else None,
            "submission_csv": str(submission_path) if submission_path else None,   # NE
            "model_name": model_name,
            "model_uri": resolved_model_uri,
            "model_version": resolved_model_version,
            "source_run_id": source_run_id,
            "db_run_id": db_prediction_run_id,
            "f1_micro": f1_micro,
            "split": split,
            "threshold": threshold,
        }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICE inference from SQL using MLflow Registry")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="val | pseudo_test | test (default: config predict_split or val)"
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=MLFLOW_CHAMPION_ALIAS,
        help=f"Registry alias to use (default: {MLFLOW_CHAMPION_ALIAS}). Example: {MLFLOW_CANDIDATE_ALIAS}"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional explicit MLflow model version. If provided, it overrides --model-alias."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MLFLOW_REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {MLFLOW_REGISTERED_MODEL_NAME})"
    )

    args = parser.parse_args()

    predict(
        threshold=args.threshold,
        batch_size=args.batch_size,
        out_path=args.output,
        split=args.split,
        model_alias=args.model_alias,
        model_version=args.model_version,
        model_name=args.model_name,
    )