# https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html
# https://auto.gluon.ai/stable/install.html

import numpy as np
import warnings
import pandas as pd
import joblib
import logging
from pathlib import Path
from mlflow import log_metric, log_param, log_artifacts

import sys
from from_root import from_root

sys.path.append(str(from_root(".")))
from utils import paths_catalog

CURRENT_DIRECTORY = Path(__file__).resolve().parent
MODEL_SAVE_PATH = paths_catalog.AUTOGLUON_MODEL
NUM_ROWS_TO_EXTRACT = 110400
NUM_TEST_ROWS = 10000


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def train_test():
    try:
        MODEL_SAVE_PATH.mkdir()
    except FileExistsError:
        logger.info(f"{MODEL_SAVE_PATH} exists, using it")

    logger.addHandler(
        logging.FileHandler(
            filename=MODEL_SAVE_PATH / "autogluon_training.log", mode="w"
        )
    )

    warnings.filterwarnings("ignore")
    np.random.seed(123)

    DATASET_NAME = "full_preprocessed_reddit_twitter_dataset.joblib"
    DATASET_LOCATION = paths_catalog.PREPROCESSED_DATASETS / DATASET_NAME
    logger.info(f"Dataset: {DATASET_LOCATION}")

    dataset_df = joblib.load(DATASET_LOCATION)[["sentence", "label"]]

    num_training_rows = NUM_ROWS_TO_EXTRACT - NUM_TEST_ROWS
    train_data = dataset_df.iloc[:num_training_rows]
    test_data = dataset_df.iloc[-NUM_TEST_ROWS:]
    logger.info(f"{num_training_rows} rows used for training")
    # mlflow tracking
    log_param("sentimental analysis sample size", num_training_rows)

    """training"""
    from autogluon.text import TextPredictor

    SECS_IN_HOUR = 3600
    SECS_IN_MIN = 60
    time_limit_in_secs = 15 * SECS_IN_HOUR
    predictor = TextPredictor(label="label", eval_metric="acc", path=MODEL_SAVE_PATH)
    predictor.fit(train_data, time_limit=time_limit_in_secs)

    """Evaluation"""
    test_score = predictor.evaluate(test_data, metrics=["acc", "f1"])
    log_metric("sentiment analysis f1", test_score)
    print(test_score)
    # mlflow tracking
    log_artifacts(MODEL_SAVE_PATH)


if __name__ == "__main__":
    train_test()
