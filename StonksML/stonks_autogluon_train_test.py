# https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html
# https://auto.gluon.ai/stable/install.html

import numpy as np
import warnings
import pandas as pd
import logging
from pathlib import Path
import math
import matplotlib.pyplot as plt
from autogluon.core.utils.loaders.load_pd import load
import joblib

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
dataset_name = "training.1600000.processed.noemoticon.csv"
dataset_location = file_dir.resolve().parent / dataset_folder / dataset_name
logger.info(f"Dataset: {dataset_location}")

dataset_df = pd.read_csv(dataset_location)
dataset_df.columns = ["target", "ids", "date", "flag", "user", "text"]

dataset_df = dataset_df[["text", "target"]]
dataset_df.columns = ["sentence", "label"]

dataset_df["label"].replace([0, 2, 4], [0, 0, 1], inplace=True)
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

num_training_rows = 250000
train_data = dataset_df.iloc[:num_training_rows]
test_data = dataset_df.iloc[-5000:]
sample_data = dataset_df.sample(n=1000)
logger.info(f"{num_training_rows} rows used for training")


train_data.to_csv(file_dir / "train_data.csv")
sample_data.to_csv(file_dir / "sample_data.csv")
test_data.to_csv(file_dir / "test_data.csv")


# train_data = load(
# "https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet"
# )

# test_data = load(
#     "https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet"
# )


"""training"""
from autogluon.text import TextPredictor

time_limit_in_secs = 60 * 60 * 15
predictor = TextPredictor(label="label", eval_metric="acc", path=model_save_path)
predictor.fit(train_data, time_limit=time_limit_in_secs)

"""Evaluation"""
test_score = predictor.evaluate(test_data, metrics=["acc", "f1"])
print(test_score)

"""Intermediate Results"""
# predictor.results.tail(3)


"""Direct prediction (Boolean)"""
sentence1 = "it's a charming and often affecting journey."
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({"sentence": [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Sentiment":', predictions[0])
print('"Sentence":', sentence2, '"Predicted Sentiment":', predictions[1])

"""Probabilistic predictions"""
probs = predictor.predict_proba({"sentence": [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Class-Probabilities":', probs[0])
print('"Sentence":', sentence2, '"Predicted Class-Probabilities":', probs[1])


# """Save and load"""
# loaded_predictor = TextPredictor.load("ag_sst")  # load automatically saved predictor"
# loaded_predictor.predict_proba({"sentence": [sentence1, sentence2]})

# """Saving to custom location"""
# loaded_predictor.save("my_saved_dir")
# loaded_predictor2 = TextPredictor.load("my_saved_dir")
# loaded_predictor2.predict_proba({"sentence": [sentence1, sentence2]})

"""Extract embedding"""
# embeddings = predictor.extract_embedding(test_data)
# print(embeddings)

# from sklearn.manifold import TSNE

# X_embedded = TSNE(n_components=2, random_state=123).fit_transform(embeddings)
# for val, color in [(0, "red"), (1, "blue")]:
#     idx = (test_data["label"].to_numpy() == val).nonzero()
#     plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=color, label=f"label={val}")
# plt.legend(loc="best")
