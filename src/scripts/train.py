import argparse
import logging
import os
import pickle
from io import BytesIO

import config_logging
import mlflow
import pandas as pd

from house_price_prediction.training_package import training

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


def model_training(
    input_data_folder,
    output_model_folder,
    data_prep_run_id='330da2288b7e46aea5f3b17730c78d40',
):
    """Method to train the model"""
    artifacts = list_artifacts(data_prep_run_id)
    artifacts = [artifact for artifact in artifacts if input_data_folder in artifact]
    for artifact in artifacts:
        if 'X_train' in artifact:
            housing_df_X = read_csv_from_artifact(data_prep_run_id, artifact)
        elif 'y_train' in artifact:
            housing_df_y = read_csv_from_artifact(data_prep_run_id, artifact)

    os.makedirs(output_model_folder, exist_ok=True)
    lin_reg_model = training.create_linear_regressor_model(housing_df_X, housing_df_y)
    with open(output_model_folder + "/linear_regresion_model.pkl", 'wb') as f:
        pickle.dump(lin_reg_model, f)
    mlflow.log_artifact(
        output_model_folder + "/linear_regresion_model.pkl",
        artifact_path=f"{output_model_folder}/linear_regression",
    )
    logger.info("Linear Regression training completed")

    tree_reg_model = training.create_decision_tree_model(housing_df_X, housing_df_y)
    with open(output_model_folder + "/decision_tree_model.pkl", 'wb') as f:
        pickle.dump(tree_reg_model, f)
    mlflow.log_artifact(
        output_model_folder + "/decision_tree_model.pkl",
        artifact_path=f"{output_model_folder}/decision_tree_regressor",
    )
    logger.info("Decision tree training completed")

    grid_search = training.create_random_forest_with_grid_search(
        housing_df_X, housing_df_y
    )
    final_model_gs = training.get_best_params_from_gs(grid_search, housing_df_X)

    with open(output_model_folder + "/gs_rf_model.pkl", 'wb') as f:
        pickle.dump(final_model_gs, f)
    mlflow.log_artifact(
        output_model_folder + "/gs_rf_model.pkl",
        artifact_path=f"{output_model_folder}/random_forest_regressor",
    )
    logger.info("Random Forest GS training completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_folder",
        help="Input dataset folder path",
        default="data/processed",
    )
    parser.add_argument(
        "--output_model_folder",
        help="Output dataset folder path",
        default="models",
    )
    parser.add_argument("--log_level", help="Log level", default="DEBUG")
    parser.add_argument("--log_path", help="Where to log", default=None)
    parser.add_argument(
        "--console_log",
        help="Whether or not to write logs to the console",
        debug=True,
    )
    args = parser.parse_args()
    initialize_logger(args.log_level, args.log_path, args.console_log)
    model_training(args.input_data_folder, args.output_model_folder)
