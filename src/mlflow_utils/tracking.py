import mlflow
from src.core.config import settings


def setup_mlflow():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)


def start_mlflow_run(run_name=None):
    setup_mlflow()
    return mlflow.start_run(run_name=run_name)