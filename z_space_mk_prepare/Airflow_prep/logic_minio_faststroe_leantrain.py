# airflow DAG
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG("rakuten_training_pipeline") as dag:

    # Task 1: Daten von MinIO in shared Volume ziehen
    fetch_data = DockerOperator(
        task_id="fetch_data",
        image="minio/mc:latest",
        command="""
            mc alias set local http://minio:9000 minioadmin minioadmin &&
            mc cp --recursive local/rakuten-images/ /cache/images/ &&
            mc cp local/rakuten-data/X_train.csv /cache/raw/ &&
            mc cp local/rakuten-data/y_train.csv /cache/raw/ &&
            mc cp local/rakuten-data/X_test.csv  /cache/raw/
        """,
        mounts=[Mount(target="/cache", source="training_cache", type="volume")],
        network_mode="rakuten2_default",  # gleiches Netzwerk wie MinIO
    )

    # Task 2: Lean Training Container — liest aus Cache
    train = DockerOperator(
        task_id="train",
        image="rakuten-training:lean",
        command="python -m src.models.train_model_final",
        mounts=[Mount(target="/app/data", source="training_cache", type="volume")],
        network_mode="rakuten2_default",
        environment={
            "IMAGE_SOURCE": "local",      # ← liest aus Cache, nicht MinIO
            "DATA_SOURCE":  "local",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        device_requests=[{"Driver": "nvidia", "Count": 1, 
                          "Capabilities": [["gpu"]]}],
    )

    fetch_data >> train  # erst Daten, dann Training
Shared Volume ist der Trick:
fetch_data Container  →  schreibt nach training_cache Volume
training Container    →  liest aus training_cache Volume
= Container-zu-Container über lokale Disk, kein MinIO-Overhead