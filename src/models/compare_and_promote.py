"""
Compare-and-promote orchestration for ICE models.

Purpose:
    - run multiple training jobs sequentially
    - each run may use a larger portion of the training data
    - evaluate every resulting registered model on the same eval split
    - compare micro-F1
    - promote the best model to MLflow alias 'champion'
    - optionally assign 'candidate' to the second-best model

Typical usage:
    python -m src.models.compare_and_promote
    python -m src.models.compare_and_promote --eval-split val
    python -m src.models.compare_and_promote --fractions 0.3 0.6 1.0
    python -m src.models.compare_and_promote --epochs 2 --batch-size 16

Docker usage:
    docker compose run --rm training \
      python -m src.models.compare_and_promote \
      --eval-split val \
      --fractions 0.3 0.6 1.0 \
      --epochs 2 \
      --batch-size 16
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Any

import src.streamlit.mlflow as mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_CANDIDATE_ALIAS,
    MINIO_ENDPOINT,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
)
from src.models.predict_model_final import predict


# ============================================================
# Environment helpers
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


def get_client() -> MlflowClient:
    configure_mlflow_s3()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# ============================================================
# Training helpers
# ============================================================

def run_training_subprocess(
    train_module: str,
    data_fraction: float,
    epochs: int | None,
    batch_size: int | None,
    extra_args: list[str] | None = None,
) -> None:
    """
    Launch one training run as a subprocess so the existing training module
    remains the single source of truth.

    Assumes the training module supports:
        --data-fraction
        --epochs
        --batch_size
    """
    cmd = [sys.executable, "-m", train_module]

    cmd += ["--data-fraction", str(data_fraction)]

    if epochs is not None:
        cmd += ["--epochs", str(epochs)]

    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]

    if extra_args:
        cmd += extra_args

    print("\n" + "=" * 90)
    print("Starting training run")
    print("Command:", " ".join(cmd))
    print("=" * 90)

    subprocess.run(cmd, check=True)


def get_latest_model_version_for_name(client: MlflowClient, model_name: str) -> str:
    """
    Return the numerically latest registered version for the model name.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No registered model versions found for '{model_name}'.")

    latest = max(versions, key=lambda mv: int(mv.version))
    return str(latest.version)


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_registered_version(
    model_name: str,
    model_version: str,
    eval_split: str,
    threshold: float | None,
    batch_size: int | None,
) -> dict[str, Any]:
    """
    Evaluate one registered model version on a labeled eval split using the
    existing predict() function, but without saving CSV/DB outputs.
    """
    result = predict(
        split=eval_split,
        model_name=model_name,
        model_version=model_version,
        model_alias=None,
        threshold=threshold,
        batch_size=batch_size,
        save_outputs=False,
        save_db=False,
    )

    f1_micro = result.get("f1_micro")
    if f1_micro is None:
        raise RuntimeError(
            f"Evaluation did not return f1_micro for model version {model_version}. "
            f"Make sure eval split '{eval_split}' has labels."
        )

    return result


# ============================================================
# Alias helpers
# ============================================================

def get_alias_version(
    client: MlflowClient,
    model_name: str,
    alias: str,
) -> str | None:
    """
    Return the version currently assigned to an alias, or None if not found.
    """
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return str(mv.version)
    except Exception:
        return None


def set_alias_safe(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    """
    Assign alias to the given version.
    """
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(version),
    )


# ============================================================
# Main orchestration
# ============================================================

def compare_and_promote(
    train_module: str,
    model_name: str,
    eval_split: str,
    fractions: list[float],
    threshold: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    extra_train_args: list[str] | None = None,
    assign_candidate: bool = True,
    compare_with_existing_champion: bool = True,
) -> dict[str, Any]:
    """
    Main orchestration flow:
      1. Run several sequential training jobs with increasing data fractions
      2. After each run, detect the latest registered model version
      3. Evaluate each version on the same eval split
      4. Optionally compare against the current champion too
      5. Pick the best model by micro-F1
      6. Update champion alias
      7. Optionally assign candidate alias to second-best
    """
    client = get_client()

    run_summaries: list[dict[str, Any]] = []
    seen_versions: set[str] = set()

    existing_champion_summary = None
    current_champion_version = None

    if compare_with_existing_champion:
        current_champion_version = get_alias_version(
            client=client,
            model_name=model_name,
            alias=MLFLOW_CHAMPION_ALIAS,
        )

        if current_champion_version is not None:
            print(f"Current champion before new runs: v{current_champion_version}")

            champion_eval = evaluate_registered_version(
                model_name=model_name,
                model_version=current_champion_version,
                eval_split=eval_split,
                threshold=threshold,
                batch_size=batch_size,
            )

            existing_champion_summary = {
                "run_index": 0,
                "data_fraction": None,
                "model_version": champion_eval["model_version"],
                "model_uri": champion_eval["model_uri"],
                "source_run_id": champion_eval["source_run_id"],
                "f1_micro": champion_eval["f1_micro"],
                "split": champion_eval["split"],
                "threshold": champion_eval["threshold"],
                "origin": "existing_champion",
            }

            run_summaries.append(existing_champion_summary)
            seen_versions.add(str(existing_champion_summary["model_version"]))

            print(
                "Existing champion evaluation: "
                f"version={existing_champion_summary['model_version']} | "
                f"f1_micro={existing_champion_summary['f1_micro']:.4f}"
            )
        else:
            print("No existing champion alias found. Proceeding with new runs only.")

    for idx, fraction in enumerate(fractions, start=1):
        print(f"\n########## RUN {idx}/{len(fractions)} | data_fraction={fraction} ##########")

        run_training_subprocess(
            train_module=train_module,
            data_fraction=fraction,
            epochs=epochs,
            batch_size=batch_size,
            extra_args=extra_train_args,
        )

        time.sleep(2)

        new_version = get_latest_model_version_for_name(client, model_name=model_name)
        print(f"Latest registered version after run {idx}: v{new_version}")

        if new_version in seen_versions:
            raise RuntimeError(
                f"Detected duplicate latest model version v{new_version}. "
                "This suggests the training run may not have registered a new model version."
            )

        eval_result = evaluate_registered_version(
            model_name=model_name,
            model_version=new_version,
            eval_split=eval_split,
            threshold=threshold,
            batch_size=batch_size,
        )

        run_summary = {
            "run_index": idx,
            "data_fraction": fraction,
            "model_version": eval_result["model_version"],
            "model_uri": eval_result["model_uri"],
            "source_run_id": eval_result["source_run_id"],
            "f1_micro": eval_result["f1_micro"],
            "split": eval_result["split"],
            "threshold": eval_result["threshold"],
            "origin": "new_run",
        }
        run_summaries.append(run_summary)
        seen_versions.add(str(run_summary["model_version"]))

        print(
            f"Finished run {idx}: "
            f"version={run_summary['model_version']} | "
            f"f1_micro={run_summary['f1_micro']:.4f}"
        )

    if not run_summaries:
        raise RuntimeError("No runs were completed.")

    ranked = sorted(
        run_summaries,
        key=lambda x: (
            x["f1_micro"],
            -1 if x["data_fraction"] is None else x["data_fraction"],
            int(x["model_version"]),
        ),
        reverse=True,
    )

    best = ranked[0]

    print("\n" + "=" * 90)
    print("Best model selected")
    print(json.dumps(best, indent=2))
    print("=" * 90)

    previous_champion_version = current_champion_version
    champion_changed = previous_champion_version != str(best["model_version"])

    set_alias_safe(
        client=client,
        model_name=model_name,
        alias=MLFLOW_CHAMPION_ALIAS,
        version=str(best["model_version"]),
    )

    candidate = None
    if assign_candidate:
        for item in ranked:
            if str(item["model_version"]) != str(best["model_version"]):
                candidate = item
                break

        if candidate is not None:
            set_alias_safe(
                client=client,
                model_name=model_name,
                alias=MLFLOW_CANDIDATE_ALIAS,
                version=str(candidate["model_version"]),
            )

    summary = {
        "previous_champion_version": previous_champion_version,
        "champion_version": str(best["model_version"]),
        "champion_changed": champion_changed,
        "best": best,
        "candidate": candidate,
        "all_runs": ranked,
    }

    print("\nFinal summary:")
    print(json.dumps(summary, indent=2))

    return summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multiple ICE runs, compare them, and promote the best model."
    )
    parser.add_argument(
        "--train-module",
        type=str,
        default="src.models.train_model_final",
        help="Python module path for the training script",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MLFLOW_REGISTERED_MODEL_NAME,
        help="Registered MLflow model name",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="val",
        help="Labeled split used for fair comparison (recommended: val)",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.3, 0.6, 1.0],
        help="Training data fractions for sequential runs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional prediction threshold for evaluation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional training epochs override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override",
    )
    parser.add_argument(
        "--no-candidate",
        action="store_true",
        help="Do not assign candidate alias to the second-best model",
    )
    parser.add_argument(
        "--skip-existing-champion",
        action="store_true",
        help="Do not include the current champion in the comparison",
    )
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Any extra args passed through to the training module",
    )

    args = parser.parse_args()

    compare_and_promote(
        train_module=args.train_module,
        model_name=args.model_name,
        eval_split=args.eval_split,
        fractions=args.fractions,
        threshold=args.threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        extra_train_args=args.extra_train_args,
        assign_candidate=not args.no_candidate,
        compare_with_existing_champion=not args.skip_existing_champion,
    )