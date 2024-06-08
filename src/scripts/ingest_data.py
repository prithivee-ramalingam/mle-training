import argparse
import logging
import os

import config_logging

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
    logger.info("Data Loaded Successfully")

    # Train and Test split the data
    strat_train_set, strat_test_set = (
        data_ingestion.perform_stratified_sampling_based_on_income_category(housing)
    )

    X_train, y_train, imputer = data_ingestion.prepare_data(strat_train_set, 'train')
    X_test, y_test, imputer = data_ingestion.prepare_data(
        strat_test_set, 'test', imputer
    )

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", help="Output Folder path")
    parser.add_argument("log_level", help="Log level")
    parser.add_argument("log_path", help="Where to log")
    parser.add_argument(
        "console_log", help="Whether or not to write logs to the console"
    )
    args = parser.parse_args()
    initialize_logger(args.log_level, args.log_path, args.console_log)
    ingest_input_data(args.output_folder)


if __name__ == '__main__':
    main()
