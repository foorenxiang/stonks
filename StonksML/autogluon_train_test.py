# https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html
# https://auto.gluon.ai/stable/install.html

import numpy as np
import warnings
import pandas as pd
import joblib
import logging
from pathlib import Path
from utils import paths_catalog

CURRENT_DIRECTORY = Path(__file__).resolve().parent
MODEL_SAVE_PATH = paths_catalog.AUTOGLUON_MODEL


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

try:
    MODEL_SAVE_PATH.mkdir()
except FileExistsError:
    logger.info(f"{MODEL_SAVE_PATH} exists, using it")

logger.addHandler(
    logging.FileHandler(filename=MODEL_SAVE_PATH / "autogluon_training.log", mode="w")
)

warnings.filterwarnings("ignore")
np.random.seed(123)

DATASET_NAME = "full_preprocessed_reddit_twitter_dataset.joblib"
DATASET_LOCATION = paths_catalog.PREPROCESSED_DATASETS / DATASET_NAME
logger.info(f"Dataset: {DATASET_LOCATION}")

dataset_df = joblib.load(DATASET_LOCATION)[["sentence", "label"]]

num_rows_to_extract = 110400
num_test_rows = 10000
num_training_rows = num_rows_to_extract - num_test_rows
train_data = dataset_df.iloc[:num_training_rows]
test_data = dataset_df.iloc[-num_test_rows:]
sample_data = dataset_df.sample(n=1000)
logger.info(f"{num_training_rows} rows used for training")


"""training"""
from autogluon.text import TextPredictor

time_limit_in_secs = 60 * 60 * 15
predictor = TextPredictor(label="label", eval_metric="acc", path=MODEL_SAVE_PATH)
predictor.fit(train_data, time_limit=time_limit_in_secs)

"""Evaluation"""
test_score = predictor.evaluate(test_data, metrics=["acc", "f1"])
print(test_score)
