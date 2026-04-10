#!/bin/bash
# start_local_dev.sh — Start MLflow locally for development
# Usage: source start_local_dev.sh

# Load .env variables into current shell
set -a
source .env
set +a

# Override for local context (Docker hostnames don't work locally)
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

echo "=== Local Dev Environment ==="
echo "  MLFLOW_TRACKING_URI:    $MLFLOW_TRACKING_URI"
echo "  MLFLOW_S3_ENDPOINT_URL: $MLFLOW_S3_ENDPOINT_URL"
echo "  AWS_ACCESS_KEY_ID:      $AWS_ACCESS_KEY_ID"
echo "  ICE_MAX_EPOCHS:         $ICE_MAX_EPOCHS"
echo ""
echo "Starting MLflow server (local, PostgreSQL disabled, SQLite fallback)..."
echo "  Press Ctrl+C to stop."
echo ""

mlflow server \
  --backend-store-uri sqlite:///./mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"