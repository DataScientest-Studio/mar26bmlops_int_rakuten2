import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten_color_classification")
    MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
    APP_ENV = os.getenv("APP_ENV", "development")


settings = Settings()