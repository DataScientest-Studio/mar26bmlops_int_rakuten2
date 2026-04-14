# airflow/dags/rakuten_incremental_training.py
"""
Airflow DAG: Incremental ICE Training Pipeline

8 sequential training runs (100k → 170k images, 3 epochs each).
After all runs: compare on val, promote best to champion, predict test, drift report.

IMPORTANT: The project has a mlflow/ folder in its root that shadows the
real mlflow package. All task functions must import mlflow BEFORE adding
/opt/airflow/app to sys.path.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

TOTAL_TRAIN = 190_908
APP_ROOT = "/opt/airflow/app"

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


def _add_app_to_path():
    """Add project root to sys.path for src.* imports.
    Call AFTER importing mlflow to avoid mlflow/ folder shadow."""
    import sys
    if APP_ROOT not in sys.path:
        sys.path.insert(0, APP_ROOT)


def task_check_prerequisites(**context):
    """Verify DB is populated and MLflow is reachable."""
    import os
    import sqlite3
    # Import mlflow FIRST (before app root is on sys.path)
    import mlflow

    # Check DB
    db_path = os.path.join(APP_ROOT, "db", "rakuten.db")
    if os.path.exists(db_path):
        conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT split, COUNT(*) FROM products GROUP BY split")
        counts = dict(cur.fetchall())
        conn.close()
        print(f"DB products (SQLite): {counts}")
        train_count = counts.get("train", 0)
        val_count = counts.get("val", 0)
    else:
        # Try Postgres via src.db
        _add_app_to_path()
        from src.db import get_product_count
        train_count = get_product_count(split="train")
        val_count = get_product_count(split="val")
        print(f"DB products (Postgres) — train: {train_count:,}  val: {val_count:,}")

    assert train_count > 100_000, f"Not enough training data: {train_count}"
    assert val_count > 10_000, f"Not enough val data: {val_count}"

    # Check MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiments = mlflow.search_experiments()
    print(f"MLflow reachable at {tracking_uri}, {len(experiments)} experiments found")

    context["ti"].xcom_push(key="run_configs", value=RUN_CONFIGS)


def task_train_run(run_index: int, data_fraction: float, epochs: int, **context):
    """Execute one training run with the given data fraction."""
    # Import mlflow FIRST to avoid shadow
    import mlflow  # noqa: F401

    _add_app_to_path()
    from src.models.train_model_final import train

    print(f"\n{'='*60}")
    print(f"TRAINING RUN {run_index}/8 — fraction={data_fraction} "
          f"(~{int(data_fraction * TOTAL_TRAIN):,} images), epochs={epochs}")
    print(f"{'='*60}")

    result = train(config={
        "data_fraction": data_fraction,
        "max_epochs": epochs,
    })

    summary = {
        "run_index": run_index,
        "data_fraction": data_fraction,
        "run_id": result["run_id"],
        "model_version": result.get("model_version"),
        "val_f1_micro": result.get("best_val_metrics", {}).get("micro_f1", 0),
    }
    context["ti"].xcom_push(key=f"run_{run_index}_result", value=summary)
    print(f"Run {run_index} done — F1={summary['val_f1_micro']:.4f}")
    return summary


def task_compare_and_promote(**context):
    """Compare all runs and promote best model to champion."""
    import os
    # Import mlflow FIRST
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    ti = context["ti"]
    results = []
    for i in range(1, 9):
        r = ti.xcom_pull(key=f"run_{i}_result", task_ids=f"train_run_{i}")
        if r:
            results.append(r)

    if not results:
        raise RuntimeError("No training results found in XCom")

    # Rank by val F1 micro
    ranked = sorted(results, key=lambda x: x.get("val_f1_micro", 0), reverse=True)
    best = ranked[0]

    print(f"\n{'='*70}")
    print(f"{'Run':>4} {'Fraction':>10} {'Images':>10} {'F1 micro':>10} {'Version':>8}")
    print("-" * 70)
    for r in ranked:
        marker = " ★" if r == best else ""
        print(f"{r['run_index']:>4} {r['data_fraction']:>10.3f} "
              f"{int(r['data_fraction'] * TOTAL_TRAIN):>10,} "
              f"{r.get('val_f1_micro', 0):>10.4f} "
              f"{str(r.get('model_version', '?')):>8}{marker}")
    print("=" * 70)

    # Promote best to champion
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "rakuten-ice-dual-encoder")

    if best.get("model_version"):
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=str(best["model_version"]),
        )
        print(f"Champion set: version {best['model_version']}")
    else:
        # Fallback: find version by run_id
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.run_id == best["run_id"]:
                client.set_registered_model_alias(
                    name=model_name, alias="champion", version=v.version,
                )
                print(f"Champion set: version {v.version} (found by run_id)")
                break

    # Second best → candidate
    if len(ranked) > 1:
        second = ranked[1]
        if second.get("model_version"):
            client.set_registered_model_alias(
                name=model_name,
                alias="candidate",
                version=str(second["model_version"]),
            )

    ti.xcom_push(key="best_model", value=best)


def task_predict_test(**context):
    """Run prediction on the test split using champion model."""
    import mlflow  # noqa: F401 — avoid shadow
    _add_app_to_path()
    from src.models.predict_model_final import predict
    predict(split="test", model_alias="champion")
    print("Test predictions saved.")


def task_drift_report(**context):
    """Generate drift report."""
    _add_app_to_path()
    from src.monitoring.drift import run_drift
    run_drift()
    print("Drift report generated.")


# ============================================================
# DAG
# ============================================================

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
        t = BashOperator(
            task_id=f"train_run_{cfg['run_index']}",
            bash_command=f"docker exec rakuten_training python -m src.models.train_model_final --epochs {cfg['epochs']} --data-fraction {cfg['data_fraction']}",
)
        # t = PythonOperator(
        #     task_id=f"train_run_{cfg['run_index']}",
        #     python_callable=task_train_run,
        #     op_kwargs={
        #         "run_index": cfg["run_index"],
        #         "data_fraction": cfg["data_fraction"],
        #         "epochs": cfg["epochs"],
        #     },
        # )
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