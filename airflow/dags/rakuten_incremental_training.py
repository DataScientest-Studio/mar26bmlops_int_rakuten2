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
    "-v /home/mirco/rakuten2/reports:/app/reports "
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


    # hooks from ice for prometheus
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    g_run_f1 = Gauge(
        "rakuten_training_run_f1",
        "Validation F1 score per training run",
        ["model_version", "run_id"],
        registry=registry,
    )
    g_duration = Gauge(
        "rakuten_training_duration_seconds",
        "Training run duration in seconds",
        ["model_version"],
        registry=registry,
    )
    g_champion_f1 = Gauge("rakuten_champion_f1", "Champion model F1 score", registry=registry)
    g_champion_version = Gauge("rakuten_champion_version", "Champion model version number", registry=registry)

    for r in results:
        g_run_f1.labels(model_version=r["version"], run_id=r["run_id"]).set(r["f1"])
        run_info = client.get_run(r["run_id"]).info
        if run_info.end_time and run_info.start_time:
            duration = (run_info.end_time - run_info.start_time) / 1000
            g_duration.labels(model_version=r["version"]).set(duration)
    g_champion_f1.set(best_f1)
    g_champion_version.set(int(best.version))

    pushgateway_url = os.getenv("PUSHGATEWAY_URL", "pushgateway:9091")
    push_to_gateway(pushgateway_url, job="rakuten_training", registry=registry)
    print(f"Metrics pushed to Pushgateway at {pushgateway_url}")


    # Connections to ice
def task_push_train_metrics(run_index: int, **context):
    """Push the latest MLflow run's training metrics to Prometheus Pushgateway.

    Called after each train_run_X so Grafana can display per-run history
    even though individual training processes are short-lived.
    """
    import os
    import mlflow
    from mlflow.tracking import MlflowClient
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Grab the most recent run from our experiment
    experiment = mlflow.get_experiment_by_name(
        os.getenv("MLFLOW_EXPERIMENT", "rakuten_ice_dual_encoder")
    )
    if experiment is None:
        print("[push] MLflow experiment not found — skipping metric push")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        print("[push] No MLflow runs found — skipping metric push")
        return

    run = runs[0]
    metrics = run.data.metrics
    params = run.data.params

    # Build an isolated registry — don't pollute global
    registry = CollectorRegistry()

    g_f1_micro = Gauge(
        "rakuten_training_val_f1_micro",
        "Val micro-F1 of a training run",
        ["run_index", "data_fraction"],
        registry=registry,
    )
    g_f1_macro = Gauge(
        "rakuten_training_val_f1_macro",
        "Val macro-F1 of a training run",
        ["run_index", "data_fraction"],
        registry=registry,
    )
    g_train_time = Gauge(
        "rakuten_training_duration_seconds",
        "How long training took",
        ["run_index", "data_fraction"],
        registry=registry,
    )

    data_fraction = params.get("data_fraction", "unknown")
    f1_micro = metrics.get("best_val_f1_micro", metrics.get("val_f1_micro", 0))
    f1_macro = metrics.get("best_val_f1_macro", metrics.get("val_f1_macro", 0))
    duration = (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else 0

    g_f1_micro.labels(run_index=str(run_index), data_fraction=data_fraction).set(f1_micro)
    g_f1_macro.labels(run_index=str(run_index), data_fraction=data_fraction).set(f1_macro)
    g_train_time.labels(run_index=str(run_index), data_fraction=data_fraction).set(duration)

    # Push — job label identifies this group of metrics in Pushgateway
    push_to_gateway(
        "pushgateway:9091",
        job=f"rakuten_training_run_{run_index}",
        registry=registry,
    )
    print(f"[push] Run {run_index}: f1_micro={f1_micro:.4f}, "
          f"f1_macro={f1_macro:.4f}, duration={duration:.1f}s")


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

    # After compare_and_promote: trigger API reload so the freshly-promoted
    # champion is served without a container restart. `|| true` at the end
    # keeps the DAG green even if the API happens to be down — reload will
    # happen on next API start anyway.

    reload_api = BashOperator(
        task_id="reload_api_champion",
        bash_command=(
            "curl -sS -X POST -m 30 "
            "http://api:8000/admin/reload "
            "-H 'Content-Type: application/json' || true"
        ),
        trigger_rule="all_done",
    )
 
    check >> train_tasks[0]
    train_tasks[-1] >> compare >> reload_api




