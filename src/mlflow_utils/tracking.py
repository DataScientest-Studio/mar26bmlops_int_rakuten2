import mlflow
import mlflow.sklearn
from src.core.config import settings


def setup_mlflow() -> None:
    """
    Configure MLflow tracking URI and experiment name.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)


def start_mlflow_run(run_name: str | None = None):
    """
    Start an MLflow run after ensuring the tracking setup is ready.
    """
    setup_mlflow()
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    """
    Log multiple parameters to the active MLflow run.
    """
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    """
    Log multiple metrics to the active MLflow run.
    """
    if metrics:
        mlflow.log_metrics(metrics)


def log_metric(metric_name: str, value: float) -> None:
    """
    Log a single metric to the active MLflow run.
    """
    mlflow.log_metric(metric_name, value)


def log_sklearn_model(model, artifact_path: str = "model") -> None:
    """
    Log a scikit-learn model as an MLflow artifact.
    """
    mlflow.sklearn.log_model(model, artifact_path)


def set_tags(tags: dict) -> None:
    """
    Log tags such as project name, data type, or model family.
    """
    if tags:
        mlflow.set_tags(tags)

def is_better_model(new_score: float, current_score: float | None) -> bool:
    if current_score is None:
        return True
    return new_score > current_score