#!/bin/bash

# Start the mlflow server
mlflow server \
    --backend-store-uri mlruns/ \
    --default-artifact-root mlruns/ \
    --host 0.0.0.0 \
    --port 5000 &

# Run the main Python script
python src/scripts/main_script.py

wait