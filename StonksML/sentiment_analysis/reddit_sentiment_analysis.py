import logging
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
MODEL_SAVE_PATH = paths_catalog.AUTOGLUON_MODEL
SENTIMENT_ANALYSIS_OUTPUT = "reddit_sentiment_analysis"
SENTIMENT_ANALYSIS_SUMMARY = "reddit_sentiment_analysis_summary"

logging.basicConfig(level=logging.INFO)
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


def reddit_sentiment_analysis():
    try:
        SAVE_DIRECTORY.mkdir()
    except FileExistsError:
        logger.info(f"{SAVE_DIRECTORY} exists, using it")

    predictor = TextPredictor.load(MODEL_SAVE_PATH)

    reddit_dump_location = paths_catalog.RAW_DATASETS / "latest_reddit_data.joblib"
    reddit_dump_df = load(reddit_dump_location)

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
    reddit_dump_df.to_csv(SAVE_DIRECTORY / f"{SENTIMENT_ANALYSIS_OUTPUT}.csv")
    dump(reddit_dump_df, SAVE_DIRECTORY / f"{SENTIMENT_ANALYSIS_OUTPUT}.joblib")
    reddit_dump_df.info()

    subreddit_overall_sentiment_scores = (
        reddit_dump_df[["Subreddit", "Sentiment_score"]].groupby(["Subreddit"]).mean()
    )
    logger.info("Subreddits overall scores:")
    print(subreddit_overall_sentiment_scores)

    subreddit_overall_sentiment_scores.to_csv(
        SAVE_DIRECTORY / f"{SENTIMENT_ANALYSIS_SUMMARY}.csv"
    )
    dump(
        subreddit_overall_sentiment_scores,
        SAVE_DIRECTORY / f"{SENTIMENT_ANALYSIS_SUMMARY}.joblib",
    )


if __name__ == "__main__":
    reddit_sentiment_analysis()
