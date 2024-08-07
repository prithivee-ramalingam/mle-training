import argparse
import logging
import os

import config_logging
import mlflow

from house_price_prediction.data_ingestion_package import data_ingestion

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

logger = logging.getLogger(__name__)


def initialize_logger(log_level, log_path, console_log):
    config_logging.setup_logging(
        log_level=log_level, log_path=log_path, console_log=console_log
    )


def ingest_input_data(output_folder):
    """Ingest the input data"""
    raw_data_path = output_folder + '/raw'
    data_ingestion.fetch_housing_data(HOUSING_URL, raw_data_path)
    # Load the data
    housing = data_ingestion.load_housing_data(raw_data_path)
    housing.to_csv(raw_data_path + '/raw_data.csv', index=False)
    mlflow.log_artifact(raw_data_path + '/raw_data.csv', artifact_path='data/raw')
    mlflow.log_param("Dataset size", len(housing))
    logger.info("Data Loaded Successfully")

    # Train and Test split the data
    strat_train_set, strat_test_set = (
        data_ingestion.perform_stratified_sampling_based_on_income_category(housing)
    )

    X_train, y_train, imputer = data_ingestion.prepare_data(strat_train_set, 'train')
    X_test, y_test, imputer = data_ingestion.prepare_data(
        strat_test_set, 'test', imputer
    )

    mlflow.log_param("Train Data size", len(X_train))
    mlflow.log_param("Test Data size", len(X_test))
    mlflow.log_metric(key="rmse", value=200)

    # Save the output to the folder
    processed_data_path = os.path.join(output_folder, 'processed')
    x_train_csv_path = os.path.join(processed_data_path, 'X_train.csv')
    y_train_csv_path = os.path.join(processed_data_path, 'y_train.csv')
    x_test_csv_path = os.path.join(processed_data_path, 'X_test.csv')
    y_test_csv_path = os.path.join(processed_data_path, 'y_test.csv')

    os.makedirs(processed_data_path, exist_ok=True)
    X_train.to_csv(x_train_csv_path, index=False)
    y_train.to_csv(y_train_csv_path, index=False)
    X_test.to_csv(x_test_csv_path, index=False)
    y_test.to_csv(y_test_csv_path, index=False)
    logger.info("Train Test dataset split Successfully and saved")

    mlflow.log_artifact(x_train_csv_path, artifact_path='data/processed')
    mlflow.log_artifact(y_train_csv_path, artifact_path='data/processed')
    mlflow.log_artifact(x_test_csv_path, artifact_path='data/processed')
    mlflow.log_artifact(y_test_csv_path, artifact_path='data/processed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_data_folder", help="Output Folder path", default="data"
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
    ingest_input_data(args.output_data_folder)
