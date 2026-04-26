#!/bin/bash
source venv/bin/activate
docker compose -f docker-compose.yml -f docker-compose.airflow.yml up