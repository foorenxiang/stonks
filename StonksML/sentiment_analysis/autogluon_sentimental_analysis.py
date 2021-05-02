import logging
from from_root.from_here import from_here
import pandas as pd
from pathlib import Path
from joblib import load, dump
from autogluon.text import TextPredictor

import sys
from from_root import from_root

sys.path.append(str(from_root(".")))
from utils import paths_catalog

CURRENT_DIRECTORY = Path(__file__).resolve().parent
SAVE_DIRECTORY = paths_catalog.PREPROCESSED_DATASETS
LOGGING_DIRECTORY = paths_catalog.AUTOGLUON_LOGS
INPUT_DATASET_NAME = "latest_reddit_data.joblib"
OUTPUT_DATASET_NAME = "reddit_sentiment_analysis"


def reddit_sentiment_analysis():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    try:
        LOGGING_DIRECTORY.mkdir()
    except FileExistsError:
        logger.info(f"{LOGGING_DIRECTORY} exists, using it")

    logger.addHandler(
        logging.FileHandler(
            filename=LOGGING_DIRECTORY / "reddit_sentiments.log",
            mode="w",
        )
    )

    try:
        SAVE_DIRECTORY.mkdir()
    except FileExistsError:
        logger.info(f"{SAVE_DIRECTORY} exists, using it")

    model_save_path = paths_catalog.AUTOGLUON_MODEL
    predictor = TextPredictor.load(model_save_path)

    reddit_dump_df = load(paths_catalog.RAW_DATASETS / INPUT_DATASET_NAME)

    sentiment_scores = predictor.predict({"sentence": reddit_dump_df["Title"]})

    sentiment_literal = sentiment_scores.replace(
        to_replace=0, value="Neutral/Negative"
    ).replace(to_replace=1, value="Positive")
    reddit_dump_df.insert(
        loc=reddit_dump_df.columns.get_loc("Title") + 1,
        column="Sentiment_score",
        value=sentiment_scores,
    )
    reddit_dump_df.insert(
        loc=reddit_dump_df.columns.get_loc("Title") + 1,
        column="Sentiment",
        value=sentiment_literal,
    )

    logger.info(reddit_dump_df.head(20))
    reddit_dump_df.to_csv(SAVE_DIRECTORY / f"{OUTPUT_DATASET_NAME}.csv")
    dump(reddit_dump_df, SAVE_DIRECTORY / f"{OUTPUT_DATASET_NAME}.joblib")


if __name__ == "__main__":
    reddit_sentiment_analysis()
