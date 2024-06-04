import argparse
import logging
import os
import pickle

import config_logging
import pandas as pd

from house_price_prediction.training_package import training

logger = logging.getLogger(__name__)


def initialize_logger(log_level, log_path, console_log):
    config_logging.setup_logging(
        log_level=log_level, log_path=log_path, console_log=console_log
    )


def model_training(input_path, output_path):
    """Method to train the model"""
    housing__df_X = pd.read_csv(os.path.join(input_path, 'X_train.csv'))
    housing_df_y = pd.read_csv(os.path.join(input_path, 'y_train.csv')).values.ravel()

    os.makedirs(output_path, exist_ok=True)

    lin_reg_model = training.create_linear_regressor_model(housing__df_X, housing_df_y)
    with open(output_path + "/linear_regresion_model.pkl", 'wb') as f:
        pickle.dump(lin_reg_model, f)
    logger.info("Linear Regression training completed")

    tree_reg_model = training.create_decision_tree_model(housing__df_X, housing_df_y)
    with open(output_path + "/decision_tree_model.pkl", 'wb') as f:
        pickle.dump(tree_reg_model, f)
    logger.info("Decision tree training completed")

    grid_search = training.create_random_forest_with_grid_search(
        housing__df_X, housing_df_y
    )
    final_model_gs = training.get_best_params_from_gs(grid_search, housing__df_X)

    with open(output_path + "/gs_cv_model.pkl", 'wb') as f:
        pickle.dump(final_model_gs, f)
    print("Random Forest GS training completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Input dataset folder path")
    parser.add_argument("output_folder", help="Output dataset folder path")
    parser.add_argument("log_level", help="Log level")
    parser.add_argument("log_path", help="Where to log")
    parser.add_argument(
        "console_log", help="Whether or not to write logs to the console"
    )
    args = parser.parse_args()
    initialize_logger(args.log_level, args.log_path, args.console_log)
    model_training(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()
