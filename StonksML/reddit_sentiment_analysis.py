import logging
import pandas as pd
from pathlib import Path
from joblib import load, dump
from autogluon.text import TextPredictor

current_directory = Path(__file__).resolve().parent
save_directory = current_directory.parent / "datasets" / "reddit_dump"

try:
    save_directory.mkdir()
except FileExistsError:
    pass

logging_directory = current_directory / "autogluon_logs"
try:
    logging_directory.mkdir()
except FileExistsError:
    pass

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            filename=logging_directory / "reddit_sentiments.log",
            mode="w",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger()


def reddit_sentiment_analysis():
    model_save_path = Path(__file__).resolve().parent / "autogluon_model"
    predictor = TextPredictor.load(model_save_path)

    current_directory = Path(__file__).resolve().parent
    reddit_dump_location = (
        current_directory.parent / "datasets" / "reddit_dump" / "reddit_dataset.joblib"
    )

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
    reddit_dump_df.to_csv(save_directory / "reddit_sentiment_analysis.csv", index=False)
    dump(reddit_dump_df, save_directory / "reddit_sentiment_analysis.joblib")


if __name__ == "__main__":
    reddit_sentiment_analysis()