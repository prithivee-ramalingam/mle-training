import argparse
import logging
import os
import pickle
from io import BytesIO

import config_logging
import mlflow
import numpy as np
import pandas as pd

from house_price_prediction.scoring_package import scoring

logger = logging.getLogger(__name__)


def initialize_logger(log_level, log_path, console_log):
    config_logging.setup_logging(
        log_level=log_level, log_path=log_path, console_log=console_log
    )


def list_artifacts(run_id):
    """List all artifacts for the given run ID."""
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifact_paths = []

    while artifacts:
        for artifact in artifacts:
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                artifacts.extend(sub_artifacts)
            else:
                artifact_paths.append(artifact.path)
        artifacts = artifacts[len(artifact_paths) :]

    return artifact_paths


def read_csv_from_artifact(run_id, artifact_path):
    """Read a CSV artifact into a pandas DataFrame."""
    client = mlflow.tracking.MlflowClient()
    artifact_uri = client.download_artifacts(run_id, artifact_path)
    with open(artifact_uri, 'rb') as f:
        data = f.read()
    return pd.read_csv(BytesIO(data))


def calculate_model_score(
    input_data_folder,
    output_model_folder,
    output_predictions_path,
    data_prep_run_id='4f0b94933cea481fb8858056e5bb4332',
    model_training_run_id='330da2288b7e46aea5f3b17730c78d40',
):
    """Calculate the model score based the prediction"""
    data_artifacts = list_artifacts(data_prep_run_id)
    data_artifacts = [
        artifact for artifact in data_artifacts if input_data_folder in artifact
    ]
    for artifact in data_artifacts:
        if 'X_test' in artifact:
            X_test = read_csv_from_artifact(data_prep_run_id, artifact)
        elif 'y_test' in artifact:
            Y_test = read_csv_from_artifact(data_prep_run_id, artifact)

    os.makedirs(output_predictions_path, exist_ok=True)
    model_artifacts = list_artifacts(model_training_run_id)
    model_artifacts = [
        artifact for artifact in model_artifacts if output_model_folder in artifact
    ]
    lin_reg_flag = 0
    dt_reg_flag = 0
    rf_reg_flag = 0
    client = mlflow.tracking.MlflowClient()
    for artifact in model_artifacts:
        print("#" * 100)
        print(artifact)
        if 'linear_regression' in artifact:
            if lin_reg_flag == 1:
                continue
            lin_reg_flag = 1

            with mlflow.start_run(nested=True) as linar_regression_run:
                mlflow.set_tag("stage", "linear_regression")
                local_path = client.download_artifacts(
                    model_training_run_id, artifact, output_model_folder
                )
                with open(local_path, 'rb') as f:
                    lin_reg = pickle.load(f)
                final_predictions, final_mse, final_rmse = scoring.get_prediction(
                    lin_reg, X_test, Y_test
                )
                np.savetxt(
                    f'{output_predictions_path}/final_predictions.csv',
                    final_predictions,
                    delimiter=",",
                    fmt='%d',
                )
                mlflow.log_metric('mse', final_mse)
                mlflow.log_metric('rmse', final_rmse)
                mlflow.log_artifact(
                    f'{output_predictions_path}/final_predictions.csv',
                    artifact_path=output_predictions_path,
                )

        elif 'decision_tree_regressor' in artifact:
            if dt_reg_flag == 1:
                continue
            dt_reg_flag = 1

            with mlflow.start_run(nested=True) as dt_run:
                mlflow.set_tag("stage", "decision_tree_regressor")
                local_path = client.download_artifacts(
                    model_training_run_id, artifact, output_model_folder
                )
                with open(local_path, 'rb') as f:
                    dt_reg = pickle.load(f)
                final_predictions, final_mse, final_rmse = scoring.get_prediction(
                    dt_reg, X_test, Y_test
                )
                np.savetxt(
                    f'{output_predictions_path}/final_predictions.csv',
                    final_predictions,
                    delimiter=",",
                    fmt='%d',
                )
                mlflow.log_metric('mse', final_mse)
                mlflow.log_metric('rmse', final_rmse)
                mlflow.log_artifact(
                    f'{output_predictions_path}/final_predictions.csv',
                    artifact_path=output_predictions_path,
                )

        elif 'random_forest_regressor' in artifact:
            if rf_reg_flag == 1:
                continue
            rf_reg_flag = 1
            with mlflow.start_run(nested=True) as rf_run:
                mlflow.set_tag("stage", "random_forest_regressor")
                local_path = client.download_artifacts(
                    model_training_run_id, artifact, output_model_folder
                )
                with open(local_path, 'rb') as f:
                    rf_reg = pickle.load(f)
                final_predictions, final_mse, final_rmse = scoring.get_prediction(
                    rf_reg, X_test, Y_test
                )
                np.savetxt(
                    f'{output_predictions_path}/final_predictions.csv',
                    final_predictions,
                    delimiter=",",
                    fmt='%d',
                )
                mlflow.log_metric('mse', final_mse)
                mlflow.log_metric('rmse', final_rmse)
                mlflow.log_artifact(
                    f'{output_predictions_path}/final_predictions.csv',
                    artifact_path=output_predictions_path,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_folder",
        help="Input Dataset Folder Path",
        default="data/processed",
    )
    parser.add_argument(
        "--output_model_folder", help="Prediction Model FolderPath", default="models"
    )
    parser.add_argument(
        "--output_predictions_path", help="Predictions file", default="predictions"
    )
    parser.add_argument("--log_level", help="Log level", default="DEBUG")
    parser.add_argument("--log_path", help="Where to log", default=None)
    parser.add_argument(
        "--console_log",
        help="Whether or not to write logs to the console",
        default=True,
    )
    args = parser.parse_args()
    initialize_logger(args.log_level, args.log_path, args.console_log)
    calculate_model_score(
        args.input_data_folder, args.output_model_folder, args.output_predictions_path
    )
