from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


#in contaienr always correct
db_url = "postgresql://postgres:postgres@postgres:5432/rakuten"

TOTAL_TRAIN = 190_908
APP_ROOT = "/opt/airflow/app"

RUN_CONFIGS = []
for i, n_images in enumerate(range(50, 400, 50)):
    fraction = round(n_images / TOTAL_TRAIN, 6)
    RUN_CONFIGS.append({
        "run_index": i + 1,
        "n_images": n_images,
        "data_fraction": fraction,
        "epochs": 1,
    })

default_args = {
    "owner": "rakuten-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Docker command template for GPU training
DOCKER_TRAIN_CMD = (
    "docker run --rm --gpus all "
    "--shm-size=4g "
    "--network rakuten2_default "
    "-v /home/mirco/rakuten2/data:/app/data "
    "-v /home/mirco/rakuten2/models:/app/models "
    "-v /home/mirco/rakuten2/db:/app/db "
    # Use host cache directly — no need to populate a named volume
    "-v /home/mirco/.cache/huggingface:/root/.cache/huggingface "
    "-w /app "
    "-e TRANSFORMERS_OFFLINE=1 "
    "-e HF_DATASETS_OFFLINE=1 "
    "-e MLFLOW_TRACKING_URI=http://mlflow:5000 "
    "-e AWS_ACCESS_KEY_ID=admin "
    "-e AWS_SECRET_ACCESS_KEY=pwd_123_SIMV "
    "-e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 "
    "-e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/rakuten "
    "-e DB_BACKEND=postgres "
    "-e IMAGE_SOURCE=local "
    "rakuten2-training "
    "python -m src.models.train_model_final "
    "--data-fraction {fraction} --epochs {epochs} "
    "--val-fraction 0.05 "
    "--skip-champion-compare"
    # 0.05 × 21212 = ~1060 val images — still representative, much faster

)


def task_check_prerequisites(**context):
    """Verify DB is populated and MLflow is reachable."""
    import os
    import sqlite3
    import mlflow

    # Check DB via Postgres
    import psycopg2
    # db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/rakuten")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT split, COUNT(*) FROM products GROUP BY split")
    counts = dict(cur.fetchall())
    conn.close()
    print(f"DB products: {counts}")
    assert counts.get("train", 0) > 100_000, f"Not enough training data"
    assert counts.get("val", 0) > 10_000, f"Not enough val data"

    # Check MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    experiments = mlflow.search_experiments()
    print(f"MLflow reachable, {len(experiments)} experiments found")


def task_compare_and_promote(**context):
    """Compare all runs and promote best model to champion."""
    import os
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "rakuten-ice-dual-encoder")

    # Get all model versions, find the 8 most recent
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError("No model versions found")

    recent = sorted(versions, key=lambda v: int(v.version), reverse=True)[:8]

    best = None
    best_f1 = -1
    results = []

    for v in recent:
        run = client.get_run(v.run_id)
        f1 = run.data.metrics.get("best_val_f1_micro", 0)
        results.append({"version": v.version, "run_id": v.run_id, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best = v

    print(f"\n{'='*60}")
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        marker = " ★" if r["version"] == best.version else ""
        print(f"  v{r['version']}  F1={r['f1']:.4f}{marker}")
    print(f"{'='*60}")

    client.set_registered_model_alias(name=model_name, alias="champion", version=best.version)
    print(f"Champion set: v{best.version} (F1={best_f1:.4f})")


with DAG(
    dag_id="rakuten_incremental_training",
    default_args=default_args,
    description="8 GPU training runs with increasing data, then champion promotion",
    schedule_interval=None,
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["rakuten", "mlops", "training", "gpu"],
    max_active_tasks=1,
) as dag:

    check = PythonOperator(
        task_id="check_prerequisites",
        python_callable=task_check_prerequisites,
    )

    # 8 GPU training runs via BashOperator → docker compose
    train_tasks = []
    for cfg in RUN_CONFIGS:
        t = BashOperator(
            task_id=f"train_run_{cfg['run_index']}",
            bash_command=DOCKER_TRAIN_CMD.format(
                fraction=cfg["data_fraction"],
                epochs=cfg["epochs"],
            ),
            cwd="/opt/airflow/app",  # project root where docker-compose.yml lives
        )
        train_tasks.append(t)

    for i in range(len(train_tasks) - 1):
        train_tasks[i] >> train_tasks[i + 1]

    compare = PythonOperator(
        task_id="compare_and_promote",
        python_callable=task_compare_and_promote,
    )

    check >> train_tasks[0]
    train_tasks[-1] >> compare




