FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY . /app

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]