import argparse

import ingest_data
import mlflow
import score
import train

remote_server_uri = "http://0.0.0.0:5000"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("mlflow_logging_experiment")


if __name__ == "__main__":
    # Start a parent MLflow run
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_data_folder", default="data", help="Output Folder path"
    )
    parser.add_argument("--log_level", default="DEBUG", help="Log level")
    parser.add_argument("--log_path", default=None, help="Where to log")
    parser.add_argument(
        "--console_log",
        default=True,
        help="Whether or not to write logs to the console",
    )
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
    parser.add_argument(
        "--output_predictions_path", help="Predictions file", default="predictions"
    )

    args = parser.parse_args()
    with mlflow.start_run() as parent_run:

        # Data Preparation
        with mlflow.start_run(nested=True) as data_prep_run:
            data_prep_run_id = data_prep_run.info.run_id
            mlflow.set_tag("stage", "data_preparation")
            ingest_data.initialize_logger(
                args.log_level, args.log_path, args.console_log
            )
            ingest_data.ingest_input_data(args.output_data_folder)

        # Model Training
        with mlflow.start_run(nested=True) as model_training_run:
            model_training_run_id = model_training_run.info.run_id
            mlflow.set_tag("stage", "model_training")
            train.initialize_logger(args.log_level, args.log_path, args.console_log)
            train.model_training(
                args.input_data_folder, args.output_model_folder, data_prep_run_id
            )

        # Model Scoring
        with mlflow.start_run(nested=True) as model_scoring_run:
            mlflow.set_tag("stage", "model_scoring")
            score.initialize_logger(args.log_level, args.log_path, args.console_log)
            score.calculate_model_score(
                args.input_data_folder,
                args.output_model_folder,
                args.output_predictions_path,
                data_prep_run_id,
                model_training_run_id,
            )
