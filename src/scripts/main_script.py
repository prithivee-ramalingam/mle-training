import subprocess

import mlflow

remote_server_uri = "http://0.0.0.0:5000"
mlflow.set_tracking_uri(remote_server_uri)


def run_script(script_name, args):
    """
    Helper function to run a script with given arguments.
    """
    cmd = ["python", script_name] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}: {result.stderr}")
    else:
        print(f"Output of {script_name}: {result.stdout}")
    return result.returncode


if __name__ == "__main__":
    # Start a parent MLflow run
    with mlflow.start_run() as parent_run:
        # Data Preparation
        with mlflow.start_run(nested=True) as data_prep_run:
            mlflow.set_tag("stage", "data_preparation")
            data_prep_args = ["data", "DEBUG", "None", "True"]
            run_script("src/scripts/ingest_data.py", data_prep_args)

        # Model Training
        with mlflow.start_run(nested=True) as model_training_run:
            mlflow.set_tag("stage", "model_training")
            model_training_args = [
                "data/processed",
                "artifacts/models",
                "DEBUG",
                "None",
                "True",
            ]
            run_script("src/scripts/train.py", model_training_args)

        # Model Scoring
        with mlflow.start_run(nested=True) as model_scoring_run:
            mlflow.set_tag("stage", "model_scoring")
            model_scoring_args = [
                "data/processed",
                "artifacts/models",
                "artifacts/scores",
                "DEBUG",
                "None",
                "True",
            ]
            run_script("src/scripts/score.py", model_scoring_args)
