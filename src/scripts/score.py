import argparse
import csv
import os
import pickle

import pandas as pd

from house_price_prediction.scoring_package import scoring


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
                print("Model Scores saved Successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_path", help="Input Dataset Folder Path")
    parser.add_argument("model_path", help="Prediction Model FolderPath")
    parser.add_argument("output_file_path", help="Model Output file")
    args = parser.parse_args()
    calculate_model_score(args.input_data_path, args.model_path, args.output_file_path)


if __name__ == "__main__":
    main()
