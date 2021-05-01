# https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html
# https://auto.gluon.ai/stable/install.html

import numpy as np
import warnings
import pandas as pd
import joblib
import logging
from pathlib import Path

file_dir = Path(__file__).resolve().parent
model_save_path = file_dir / "autogluon_model"
try:
    model_save_path.mkdir()
except FileExistsError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            filename=model_save_path / "autogluon_training.log", mode="w"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__file__)


warnings.filterwarnings("ignore")
np.random.seed(123)

dataset_folder = "datasets"
dataset_name = "full_preprocessed_reddit_twitter_dataset.joblib"
dataset_location = (
    file_dir.resolve().parent / dataset_folder / "preprocessed" / dataset_name
)
logger.info(f"Dataset: {dataset_location}")

dataset_df = joblib.load(dataset_location)[["sentence", "label"]]

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
predictor = TextPredictor(label="label", eval_metric="acc", path=model_save_path)
predictor.fit(train_data, time_limit=time_limit_in_secs)

"""Evaluation"""
test_score = predictor.evaluate(test_data, metrics=["acc", "f1"])
print(test_score)
