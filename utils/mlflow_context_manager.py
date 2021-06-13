from pathlib import Path
from shutil import rmtree
import mlflow
from functools import wraps
from datetime import datetime


def mlflow_context_manager(experiment_name=None, run_name_prefix=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if experiment_name:
                mlflow.set_experiment(experiment_name)
                print(f"Running experiment {experiment_name}")

            def run_experiment(*args, **kwargs):
                print(f"Started mlflow run with id {run.info.run_id}")
                mlflow.autolog(log_input_examples=True, log_models=True)
                print("Started mlflow autologging")
                return func(*args, **kwargs)

            if run_name_prefix:
                with mlflow.start_run(
                    run_name=f"{run_name_prefix}_{datetime.now()}"
                ) as run:
                    result = run_experiment(*args, **kwargs)
                return result

            with mlflow.start_run() as run:
                return run_experiment(*args, **kwargs)

        return wrapper

    return decorator
