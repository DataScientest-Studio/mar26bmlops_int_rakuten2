#!/bin/bash
pkill -f "mlflow server"
pkill -f "mlflow.server:app"
echo "MLflow processes stopped."