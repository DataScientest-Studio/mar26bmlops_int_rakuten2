# airflow/dags/rakuten_incremental_training.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

TOTAL_TRAIN = 190_908

RUN_CONFIGS = []
for i, n_images in enumerate(range(100_000, 180_000, 10_000)):
    fraction = round(n_images / TOTAL_TRAIN, 3)
    RUN_CONFIGS.append({
        "run_index": i + 1,
        "n_images": n_images,
        "data_fraction": fraction,
        "epochs": 3,
    })

default_args = {
    "owner": "rakuten-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_check_prerequisites(**context):
    """Verify DB is populated and MLflow is reachable."""
    import mlflow
    import sqlite3
    import os

    # Check DB
    db_path = "/opt/airflow/app/db/rakuten.db"
    conn = sqlite3.connect(db_path)
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

    context["ti"].xcom_push(key="run_configs", value=RUN_CONFIGS)


def task_train_run(run_index: int, data_fraction: float, epochs: int, **context):
    """Execute one training run with the given data fraction."""
    import sys
    sys.path.insert(0, "/opt/airflow/app")

    print(f"\n{'='*60}")
    print(f"TRAINING RUN {run_index}/8 — fraction={data_fraction} "
          f"(~{int(data_fraction * TOTAL_TRAIN):,} images), epochs={epochs}")
    print(f"{'='*60}")

    from src.config import ICE_CONFIG
    from src.models.train_model_ice_mk import train

    # Override config for this run
    config = {
        **ICE_CONFIG,
        "max_epochs": epochs,
        "val_ratio": data_fraction,   # use fraction as val split proxy
    }

    classifier, dual_encoder, mlb, run_id = train(config=config)

    summary = {
        "run_index": run_index,
        "data_fraction": data_fraction,
        "run_id": run_id,
    }
    context["ti"].xcom_push(key=f"run_{run_index}_result", value=summary)
    print(f"Run {run_index} done — MLflow run_id: {run_id}")
    return summary


def task_compare_and_promote(**context):
    """Compare all runs and promote best model to champion."""
    import sys
    sys.path.insert(0, "/opt/airflow/app")

    from mlflow.tracking import MlflowClient
    import os

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    ti = context["ti"]
    results = []
    for i in range(1, 9):
        r = ti.xcom_pull(key=f"run_{i}_result", task_ids=f"train_run_{i}")
        if r:
            results.append(r)

    if not results:
        raise RuntimeError("No training results found in XCom")

    # Find best run by val_f1_micro from MLflow
    best = None
    best_f1 = -1
    for r in results:
        run = client.get_run(r["run_id"])
        f1 = run.data.metrics.get("best_val_f1_micro", 0)
        r["val_f1_micro"] = f1
        if f1 > best_f1:
            best_f1 = f1
            best = r

    print(f"\nBest run: {best['run_index']} — F1={best_f1:.4f} — run_id={best['run_id']}")

    # Find model version for best run
    versions = client.search_model_versions("name='rakuten-ice-dual-encoder'")
    for v in versions:
        if v.run_id == best["run_id"]:
            client.set_registered_model_alias(
                name="rakuten-ice-dual-encoder",
                alias="champion",
                version=v.version,
            )
            print(f"Champion set: version {v.version}")
            break

    ti.xcom_push(key="best_run_id", value=best["run_id"])


def task_predict_test(**context):
    """Run prediction on the test split using champion model."""
    import sys
    sys.path.insert(0, "/opt/airflow/app")

    from src.models.predict_model_ice_mk import predict
    predict(split="test")
    print("Test predictions saved.")


def task_drift_report(**context):
    """Generate drift report."""
    import sys
    sys.path.insert(0, "/opt/airflow/app")

    from src.monitoring.drift import run_drift
    run_drift()
    print("Drift report generated.")


with DAG(
    dag_id="rakuten_incremental_training",
    default_args=default_args,
    description="8 sequential ICE training runs, then champion promotion",
    schedule_interval=None,
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["rakuten", "mlops", "training"],
    max_active_tasks=1,
) as dag:

    check = PythonOperator(
        task_id="check_prerequisites",
        python_callable=task_check_prerequisites,
    )

    train_tasks = []
    for cfg in RUN_CONFIGS:
        t = PythonOperator(
            task_id=f"train_run_{cfg['run_index']}",
            python_callable=task_train_run,
            op_kwargs={
                "run_index": cfg["run_index"],
                "data_fraction": cfg["data_fraction"],
                "epochs": cfg["epochs"],
            },
        )
        train_tasks.append(t)

    for i in range(len(train_tasks) - 1):
        train_tasks[i] >> train_tasks[i + 1]

    compare = PythonOperator(
        task_id="compare_and_promote",
        python_callable=task_compare_and_promote,
    )

    predict_test = PythonOperator(
        task_id="predict_test",
        python_callable=task_predict_test,
    )

    drift = PythonOperator(
        task_id="drift_report",
        python_callable=task_drift_report,
    )

    check >> train_tasks[0]
    train_tasks[-1] >> compare >> predict_test >> drift