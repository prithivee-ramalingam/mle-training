import argparse
import logging
import os
import tarfile

import config_logging
import numpy as np
import pandas as pd
from six.moves import urllib

from house_price_prediction.data_ingestion_package import data_ingestion
from house_price_prediction.training_package import training

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
    os.makedirs(raw_data_path, exist_ok=True)
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    data_ingestion.fetch_housing_data(HOUSING_URL, raw_data_path)
    # Load the data
    housing = data_ingestion.load_housing_data(raw_data_path)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    logger.info("Data Loaded Successfully")
    # Train and Test split the data
    train_set, test_set, train, test = training.stratifiedShuffleSplit(housing)
    # Pre process the data
    housing, y_train, X_train = data_ingestion.impute_data(train)
    X_train = training.get_household_related_information(X_train)

    X_cat = train[["ocean_proximity"]]
    X_train = X_train.join(pd.get_dummies(X_cat, drop_first=True))

    housing, y_test, X_test = data_ingestion.impute_data(test)
    X_test = training.get_household_related_information(X_test)

    X_cat = housing[["ocean_proximity"]]
    X_test = X_test.join(pd.get_dummies(X_cat, drop_first=True))

    # Save the output to the folder
    processed_data_path = os.path.join(output_folder, 'processed')
    x_train_csv_path = os.path.join(processed_data_path, 'X_train.csv')
    y_train_csv_path = os.path.join(processed_data_path, 'y_train.csv')
    x_test_csv_path = os.path.join(processed_data_path, 'X_test.csv')
    y_test_csv_path = os.path.join(processed_data_path, 'y_test.csv')

    os.makedirs(processed_data_path, exist_ok=True)
    X_train = X_train.to_csv(x_train_csv_path, index=False)
    y_train = y_train.to_csv(y_train_csv_path, index=False)
    X_test = X_test.to_csv(x_test_csv_path, index=False)
    y_test = y_test.to_csv(y_test_csv_path, index=False)
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
