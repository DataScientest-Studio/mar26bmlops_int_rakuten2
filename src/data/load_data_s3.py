# src.data.load_data_s3.py
"""
Data loading with local / MinIO toggle.

Reads CSVs either from data/raw/ (local) or from a MinIO bucket (S3),
controlled by IMAGE_SOURCE in .env:
    IMAGE_SOURCE=local  → pd.read_csv("data/raw/X_train.csv")
    IMAGE_SOURCE=minio  → stream from s3://MINIO_BUCKET_DATA/X_train.csv

Merged from VR branch (read_csv_from_minio, load_all_data_from_minio).
"""

import os
import sys
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    DATA_DIR,
    DATA_SOURCE,
    MINIO_ENDPOINT,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
    MINIO_BUCKET_DATA,
    MINIO_X_TRAIN_KEY,
    MINIO_Y_TRAIN_KEY,
    MINIO_X_TEST_KEY,
)


# ============================================================
# MinIO helpers  (from VR branch)
# ============================================================

_S3_CLIENT = None


def get_minio_client():
    """Singleton boto3 S3 client for MinIO."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )
    return _S3_CLIENT


def read_csv_from_minio(bucket: str, key: str) -> pd.DataFrame:
    """Read a CSV object from MinIO into a DataFrame."""
    s3 = get_minio_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


# ============================================================
# Local helpers
# ============================================================

def read_csv_local(filename: str) -> pd.DataFrame:
    """Read a CSV from data/raw/."""
    return pd.read_csv(DATA_DIR / "raw" / filename)


# ============================================================
# Public API — toggle via IMAGE_SOURCE
# ============================================================

def load_all_data(source: str | None = None):
    """
    Load train/test CSVs from local disk or MinIO.

    Args:
        source: "local" or "minio". Defaults to IMAGE_SOURCE from .env.

    Returns:
        (df_x, df_y, df_test) — raw DataFrames, no preprocessing.
    """
    source = (source or DATA_SOURCE).lower()

    if source == "minio":
        print("\n[Data] Loading CSVs from MinIO...")
        print(f"  Bucket: {MINIO_BUCKET_DATA}")
        print(f"  X_train: {MINIO_X_TRAIN_KEY}")
        print(f"  y_train: {MINIO_Y_TRAIN_KEY}")
        print(f"  X_test : {MINIO_X_TEST_KEY}")

        df_x = read_csv_from_minio(MINIO_BUCKET_DATA, MINIO_X_TRAIN_KEY)
        df_y = read_csv_from_minio(MINIO_BUCKET_DATA, MINIO_Y_TRAIN_KEY)
        df_test = read_csv_from_minio(MINIO_BUCKET_DATA, MINIO_X_TEST_KEY)
    else:
        print("\n[Data] Loading CSVs from local disk...")
        print(f"  Path: {DATA_DIR / 'raw'}")

        df_x = read_csv_local(MINIO_X_TRAIN_KEY)
        df_y = read_csv_local(MINIO_Y_TRAIN_KEY)
        df_test = read_csv_local(MINIO_X_TEST_KEY)

    print(f"  Train: {len(df_x):,} | Labels: {len(df_y):,} | Test: {len(df_test):,}")
    return df_x, df_y, df_test


if __name__ == "__main__":
    # Quick test: python -m src.data.load_data_s3
    df_x, df_y, df_test = load_all_data()
    print(f"\nX_train columns: {df_x.columns.tolist()}")
    print(f"y_train columns: {df_y.columns.tolist()}")
    print(f"X_test  columns: {df_test.columns.tolist()}")