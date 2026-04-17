FROM apache/airflow:2.10.4-python3.11

# Create docker group matching host GID so socket access works
# GID is injected at build time via --build-arg
ARG DOCKER_GID=999
USER root
RUN groupadd -for -g ${DOCKER_GID} docker \
 && usermod -aG docker airflow
USER airflow

RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
    mlflow==2.12.1 \
    boto3 \
    psycopg2-binary \
    scikit-learn \
    "numpy==1.26.4" \
    pillow \
    tqdm \
    pyyaml \
    python-dotenv \
    docker \
    transformers \
    fugashi \
    unidic-lite