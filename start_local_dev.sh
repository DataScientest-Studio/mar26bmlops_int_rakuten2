#!/bin/bash
# start_local_dev.sh — Start local dev environment for WSL
#
# Usage:
#   source start_local_dev.sh          # loads env + starts MLflow
#   source start_local_dev.sh --no-mlflow   # loads env only
#
# Prerequisites:
#   - venv activated
#   - Docker running (MinIO + Postgres containers)

set -a
source .env
source .env.local
set +a

echo "=== Local Dev Environment ==="
echo "  MLFLOW_TRACKING_URI:    $MLFLOW_TRACKING_URI"
echo "  MLFLOW_S3_ENDPOINT_URL: $MLFLOW_S3_ENDPOINT_URL"
echo "  MINIO_ENDPOINT:         $MINIO_ENDPOINT"
echo "  IMAGE_SOURCE:           $IMAGE_SOURCE"
echo "  DATA_SOURCE:            $DATA_SOURCE"
echo "  DB_BACKEND:             $DB_BACKEND"
echo ""

if [[ "$1" == "--no-mlflow" ]]; then
    echo "Environment loaded. MLflow not started (use Docker MLflow or start manually)."
    return 0 2>/dev/null || exit 0
fi

echo "Starting MLflow 2.12.1 server (local, SQLite backend)..."
echo "  UI: http://localhost:5000"
echo "  Press Ctrl+C to stop."
echo ""

mlflow server \
  --backend-store-uri sqlite:///./mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000
