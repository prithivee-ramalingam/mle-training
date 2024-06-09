import argparse
import csv
import logging
import os
import pickle

import config_logging
import mlflow
import pandas as pd

from house_price_prediction.scoring_package import scoring

logger = logging.getLogger(__name__)


def initialize_logger(log_level, log_path, console_log):
    config_logging.setup_logging(
        log_level=log_level, log_path=log_path, console_log=console_log
    )


def calculate_model_score(input_data_path, model_path, output_file_path):
    """Calculate the model score based the prediction"""
    X_test = pd.read_csv(os.path.join(input_data_path, 'X_test.csv'))
    Y_test = pd.read_csv(os.path.join(input_data_path, 'y_test.csv'))
    os.makedirs(output_file_path, exist_ok=True)
    files = os.listdir(model_path)
    for file in files:
        if os.path.isfile(os.path.join(model_path, file)):
            with open(os.path.join(model_path, file), 'rb') as f:
                final_model = pickle.load(f)
            final_predictions, final_mse, final_rmse = scoring.get_prediction(
                final_model, X_test, Y_test
            )
            csv_file_path = os.path.join(output_file_path, file[:-4] + "_score.csv")
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write the header row
                writer.writerow(["Metric", "Value"])
                # Write the RMSE and MAE values
                writer.writerow(["RMSE", final_rmse])
                writer.writerow(["MAE", final_mse])
                logger.info("Model Scores saved Successfully")
            mlflow.log_artifact(output_file_path, file[:-4] + "_score.csv")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("input_data_path", help="Input Dataset Folder Path")
#     parser.add_argument("model_path", help="Prediction Model FolderPath")
#     parser.add_argument("output_file_path", help="Model Output file")
#     parser.add_argument("log_level", help="Log level")
#     parser.add_argument("log_path", help="Where to log")
#     parser.add_argument(
#         "console_log", help="Whether or not to write logs to the console"
#     )
#     args = parser.parse_args()
#     initialize_logger(args.log_level, args.log_path, args.console_log)
#     with mlflow.start_run() as run:
#         calculate_model_score(
#             args.input_data_path, args.model_path, args.output_file_path
#         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_path", help="Input Dataset Folder Path")
    parser.add_argument("model_path", help="Prediction Model FolderPath")
    parser.add_argument("output_file_path", help="Model Output file")
    parser.add_argument("log_level", help="Log level")
    parser.add_argument("log_path", help="Where to log")
    parser.add_argument(
        "console_log", help="Whether or not to write logs to the console"
    )
    args = parser.parse_args()
    initialize_logger(args.log_level, args.log_path, args.console_log)
    # with mlflow.start_run() as run:
    calculate_model_score(args.input_data_path, args.model_path, args.output_file_path)
