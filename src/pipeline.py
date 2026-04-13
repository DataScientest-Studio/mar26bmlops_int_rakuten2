"""
Main pipeline: one call runs everything.

Usage:
    python -m src.pipeline --mode ingest
    python -m src.pipeline --mode ingest --mission_mode
    python -m src.pipeline --mode train
    python -m src.pipeline --mode predict
    python -m src.pipeline --mode full
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

import boto3
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ICE_CONFIG,
    MLFLOW_EXPERIMENT,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET_DATA,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_X_TEST_KEY,
    MINIO_X_TRAIN_KEY,
    MINIO_Y_TRAIN_KEY,
    export_params,
)
from src.db import clear_products, get_db_summary, ingest_products, init_db
from src.models.predict_model_ice_mk import predict
from src.models.train_model_ice_mk import train


export_params()  # sync params.yaml before every run


def get_minio_client():
    """Create MinIO/S3 client."""
    endpoint = MINIO_ENDPOINT
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = f"http://{endpoint}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def read_csv_from_minio(bucket: str, key: str) -> pd.DataFrame:
    """Read a CSV object from MinIO into a DataFrame."""
    s3 = get_minio_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


def load_all_data_from_minio():
    """Load train/test CSVs from MinIO."""
    print("\n[1/6] Loading data from MinIO...")
    print(f"  Endpoint: {MINIO_ENDPOINT}")
    print(f"  Bucket: {MINIO_BUCKET_DATA}")
    print(f"  X_train: {MINIO_X_TRAIN_KEY}")
    print(f"  y_train: {MINIO_Y_TRAIN_KEY}")
    print(f"  X_test : {MINIO_X_TEST_KEY}")

    df_x = read_csv_from_minio(MINIO_BUCKET_DATA, MINIO_X_TRAIN_KEY)
    df_y = read_csv_from_minio(MINIO_BUCKET_DATA, MINIO_Y_TRAIN_KEY)
    df_test = read_csv_from_minio(MINIO_BUCKET_DATA, MINIO_X_TEST_KEY)

    print(f"  Train rows: {len(df_x)}")
    print(f"  Label rows: {len(df_y)}")
    print(f"  Test rows : {len(df_test)}")
    return df_x, df_y, df_test


def ingest_into_db(df_x, df_y, df_test, mission_mode=False):
    """Initialize and populate DB."""
    train_x, val_x, train_y, val_y = None, None, None, None

    if not mission_mode:
        print("\n[2/6] Train/Val split...")
        train_x, val_x, train_y, val_y = train_test_split(
            df_x,
            df_y,
            test_size=ICE_CONFIG.get("val_ratio", 0.1),
            random_state=42,
            shuffle=True,
        )
        print(f"  Train={len(train_x)}, Val={len(val_x)}")
    else:
        print("\n[2/6] Mission mode — no split, all data used for training")

    print("\n[3/6] Filling database...")
    init_db()
    clear_products()

    if mission_mode:
        ingest_products(df_x, df_y, split="train")
        ingest_products(df_test, df_y=None, split="test")
    else:
        ingest_products(train_x, train_y, split="train")
        ingest_products(val_x, val_y, split="val")
        ingest_products(df_test, df_y=None, split="test")

    summary = get_db_summary()
    print(f"  DB summary: {summary}")

    return summary


def maybe_copy_mirco_db():
    """Kept only for Mirco's local setup."""
    if os.getenv("USER") == "mirco":
        import shutil
        shutil.copy(
            "/home/mirco/rakuten2/db/rakuten_colors.db",
            "/mnt/c/02_Project_MLOPS/rakuten_colors.db"
        )
        print("DB Copy to local for Mirco only")

    python -m src.pipeline --mode ingest              # DB ingest only (dev)
    python -m src.pipeline --mode ingest --mission_mode  # DB ingest (all data)
    python -m src.pipeline --mode train               # Training only
    python -m src.pipeline --mode predict             # Prediction only
    python -m src.pipeline --mode full                # Full run
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import export_params
export_params() # sync params.yaml before every run

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR, ICE_CONFIG, MLFLOW_EXPERIMENT
)
from src.db import init_db, ingest_products, get_db_summary, clear_products
from src.models.train_model_final import train
from src.models.predict_model_final import predict
from src.data.load_data_s3 import load_all_data




def run_pipeline(mode="full", real=False, mission_mode=False, config_overrides=None):
    """
    Runs the full pipeline.

    Args:
        mode:             'full', 'ingest', 'train', 'predict'
        real:             kept for compatibility
        mission_mode:     True = use all training data
        real:             True = real Rakuten data, False = mock data
        mission_mode:     True = use all training data (Rakuten Challenge)
        config_overrides: Dict with config overrides
    """
    print("=" * 60)
    print("RAKUTEN COLOR EXTRACTION PIPELINE")
    print(f"  Mode: {mode} | Data: MinIO | Mission: {mission_mode}")
    print("=" * 60)

    df_x, df_y, df_test = None, None, None
    db_summary = None
    train_result = None
    predict_result = None
    run_id = None

    # Robust approach: always ingest from MinIO before downstream steps
    df_x, df_y, df_test = load_all_data_from_minio()
    db_summary = ingest_into_db(df_x, df_y, df_test, mission_mode=mission_mode)

    maybe_copy_mirco_db()

    if mode == "ingest":
        print("\nDone (ingest only).")
        return {
            "mode": "ingest",
            "status": "success",
            "db_summary": db_summary,
        }

    # -- 4. Training --
    if mode in ("full", "train"):
        print("\n[4/6] ICE DualEncoder training...")
        config = {**ICE_CONFIG, **(config_overrides or {})}
        train_result = train(config=config)
        run_id = train_result.get("run_id")

        print("\nTraining result summary:")
        print(f"  Run ID: {train_result.get('run_id')}")
        print(f"  Model URI: {train_result.get('model_uri')}")
        print(f"  Registered model: {train_result.get('registered_model_name')}")
        print(f"  Model version: {train_result.get('model_version')}")
        print(f"  Promote new model: {train_result.get('promote_new_model')}")

    if mode == "train":
        print("\nDone (train only).")
        return {
            "mode": "train",
            "status": "success",
            "db_summary": db_summary,
            "train_result": train_result,
        }

    # -- 5. Test prediction --
    print("\n[5/6] Predicting test set...")
    out_path = Path("data") / "submissions" / "y_pred_test.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predict_result = predict(
        split="test",
        out_path=str(out_path),
    )
    print(f"  Prediction output saved to: {out_path}")

    if mode == "predict":
        print("\nDone (predict only).")
        return {
            "mode": "predict",
            "status": "success",
            "db_summary": db_summary,
            "prediction_result": predict_result,
            "prediction_path": str(out_path),
        }

    # -- 6. Submission --
    print("\n[6/6] Creating submission...")
    sub_path = Path("data") / "submissions" / "submission_ice_v1.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {out_path}. "
            "Make sure prediction step completed successfully."
        )

    results = pd.read_csv(out_path)
    results.to_csv(sub_path, index=False)

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="ice_dual_encoder_full_pipeline"):
            mlflow.log_param("mission_mode", mission_mode)
            mlflow.log_param("mode", mode)
            if run_id is not None:
                mlflow.log_param("training_run_id", run_id)
            mlflow.log_artifact(str(sub_path))
    except Exception as e:
        print(f"  MLflow logging skipped: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE DONE!")
    print(f"  Submission: {sub_path}")
    print("=" * 60)

    return {
        "mode": mode,
        "status": "success",
        "db_summary": db_summary,
        "train_result": train_result,
        "prediction_result": predict_result,
        "prediction_path": str(out_path),
        "submission_path": str(sub_path),
    }
    print(f"  Mode: {mode} | Data: {'real' if real else 'mock'} | Mission: {mission_mode}")
    print("=" * 60)

    # -- 1. Load data -- (local or Minio, controlled by IMAGE_SOURCE)
    df_x, df_y, df_test = load_all_data()

    # -- 2. Train/Val split (dev mode only) --
    if not mission_mode:
        print("\n[2/6] Train/Val split...")
        train_x, val_x, train_y, val_y = train_test_split(
            df_x, df_y, test_size=0.1, random_state=42
        )
        print(f"  Train={len(train_x)}, Val={len(val_x)}")
    else:
        print("\n[2/6] Mission mode — no split, all data used for training")

    # -- 3. Fill database --
    if mode in ("full", "ingest"):
        print("\n[3/6] Filling database...")
        init_db()
        clear_products()
        if mission_mode:
            # All labeled data for training
            ingest_products(df_x,    df_y,     split="train")
            ingest_products(df_test, df_y=None, split="test")
        else:
            # Dev mode: 90/10 split
            ingest_products(train_x, train_y, split="train")
            ingest_products(val_x,   val_y,   split="val")
            ingest_products(df_test, df_y=None, split="test")

        summary = get_db_summary()
        print(f"  DB: {summary['products_by_split']}")

    # import os
    # if os.getenv("USER") == "mirco":
    #     import shutil
    #     shutil.copy(
    #         "/home/mirco/rakuten2/db/rakuten.db",
    #         "/mnt/c/02_Project_MLOPS/rakuten_colors.db"
    #     )
    #     print("DB Copy to local for Mirco only")

    # if mode == "ingest":
    #     print("\nDone (ingest only).")
    #     return

    # -- 4. ICE DualEncoder training --
    if mode in ("full", "train"):
        print("\n[4/6] ICE DualEncoder training...")
        config = {**ICE_CONFIG, **(config_overrides or {})}
        result = train(config=config)          # dict
        run_id = result["run_id"]              # if needed

    if mode == "train":
        print("\nDone (train only).")
        return result

    # -- 5. Test set prediction --
    if mode in ("full", "predict"):
        print("\n[5/6] Predicting test set...")
        out_path = DATA_DIR / "submissions" / "y_pred_test.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        predict(split="test", out_path=str(out_path))

    # -- 6. Submission --
    if mode == "full":
        print("\n[6/6] Creating submission...")
        sub_path = DATA_DIR / "submissions" / "submission_ice_v1.csv"
        results = pd.read_csv(out_path)
        results.to_csv(sub_path, index=False)

        try:
            import mlflow
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name="ice_dual_encoder_full"):
                mlflow.log_param("mission_mode", mission_mode)
                mlflow.log_artifact(str(sub_path))
        except Exception:
            pass

        print("\n" + "=" * 60)
        print("PIPELINE DONE!")
        print(f"  Submission: {sub_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rakuten Color Pipeline")
    parser.add_argument(
        "--mode",
        default="ingest",
        choices=["full", "ingest", "train", "predict"]
    )
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--mission_mode",
        action="store_true",
        help="Mission mode: use all training data"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["max_epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.encoder_lr is not None:
        overrides["encoder_lr"] = args.encoder_lr
    parser.add_argument("--mode", default="ingest",                                                                         # mk Change here pipline mode for more then just DB creation! for now ingest only
                        choices=["full", "ingest", "train", "predict"])
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--mission_mode", action="store_true",                                                              # switch for  store_true -> split xtrain into 80 /10 val /10 pseudo_test mode 
                        help="Mission mode: use all training data (Rakuten Challenge)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr",     type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs: overrides["max_epochs"] = args.epochs
    if args.lr:     overrides["learning_rate"] = args.lr

    run_pipeline(
        mode=args.mode,
        real=args.real,
        mission_mode=args.mission_mode,
        config_overrides=overrides,
    )
        config_overrides=overrides
    )



# Space mk: Dev
# cp /home/mirco/rakuten2/db/rakuten_colors.db /mnt/c/02_Project_MLOPS/ 
# mlflow server \
# --backend-store-uri mlruns \
# --default-artifact-root mlruns \
# --host 0.0.0.0 \
# --port 5000
