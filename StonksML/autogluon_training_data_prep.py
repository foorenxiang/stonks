import logging
from pathlib import Path
import os
import joblib
import numexpr
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentalAnalysisDataPreprocessor:
    __current_directory = Path(__file__).resolve().parent
    __datasets_directory = __current_directory.parent / "datasets"
    __training_datasets_name = {
        "training.1600000.processed.noemoticon": r"training.1600000.processed.noemoticon.csv",
        "Reddit_Data": r"Reddit_Data.csv",
        "Twitter_Data": r"Twitter_Data.csv",
    }
    __unit_rows_to_take_per_dataset = int(36800 / 2)
    __sources = {"twitter": "twitter", "reddit": "reddit"}

    @staticmethod
    def __process_shuffle_rows(df):
        return df.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def __process_annotate_source(df, source):
        df["source"] = source
        return df

    @staticmethod
    def __df_describe(df, df_name=""):
        logger.info(f"Describing dataframe {df_name}:")
        logger.info(df.info())
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns}")
        logger.info(f"Sample: {df.sample(frac=1)}")

    @classmethod
    def process_36800_reddit_posts(cls):
        def process_convert_to_positive_vs_rest_annotation(df):
            df["label"].replace([-1, 0, 1], [0, 0, 1], inplace=True)
            return df

        dataset_name = cls.__training_datasets_name["Reddit_Data"]
        dataset = {
            "name": dataset_name,
            "location": cls.__datasets_directory / dataset_name,
            "drop_duplicates": True,
            "samples_to_take": cls.__unit_rows_to_take_per_dataset * 2,
        }
        dataset["df"] = (
            pd.read_csv(dataset["location"])
            .drop_duplicates()
            .sample(dataset["samples_to_take"])
        )
        dataset["df"].columns = ["sentence", "label"]
        dataset["df"] = process_convert_to_positive_vs_rest_annotation(dataset["df"])
        dataset["df"] = cls.__process_shuffle_rows(dataset["df"])
        dataset["df"] = cls.__process_annotate_source(
            dataset["df"], cls.__sources["reddit"]
        )
        cls.__df_describe(dataset["df"], dataset["name"])
        return dataset

    @classmethod
    def process_186800_tweets(cls):
        def process_convert_to_positive_vs_rest_annotation(df):
            df["label"].replace([-1, 0, 1], [0, 0, 1], inplace=True)
            return df

        dataset_name = cls.__training_datasets_name["Twitter_Data"]
        dataset = {
            "name": dataset_name,
            "location": cls.__datasets_directory / dataset_name,
            "drop_duplicates": True,
            "samples_to_take": cls.__unit_rows_to_take_per_dataset * 1,
        }
        dataset["df"] = (
            pd.read_csv(dataset["location"])
            .drop_duplicates()
            .dropna()
            .sample(dataset["samples_to_take"])
        )
        dataset["df"].columns = ["sentence", "label"]
        dataset["df"]["label"] = dataset["df"]["label"].astype("int64")
        dataset["df"] = process_convert_to_positive_vs_rest_annotation(dataset["df"])
        dataset["df"] = cls.__process_shuffle_rows(dataset["df"])
        dataset["df"] = cls.__process_annotate_source(
            dataset["df"], cls.__sources["twitter"]
        )
        cls.__df_describe(dataset["df"], dataset["name"])
        return dataset

    @classmethod
    def process_16_million_tweets(cls):
        def process_convert_to_positive_vs_rest_annotation(df):
            df["label"].replace([0, 2, 4], [0, 0, 1], inplace=True)
            return df

        dataset_name = cls.__training_datasets_name[
            "training.1600000.processed.noemoticon"
        ]
        dataset = {
            "name": dataset_name,
            "location": cls.__datasets_directory / dataset_name,
            "drop_duplicates": True,
            "samples_to_take": cls.__unit_rows_to_take_per_dataset * 1,
        }
        dataset["df"] = (
            pd.read_csv(dataset["location"])
            .drop_duplicates()
            .sample(dataset["samples_to_take"])
        )
        dataset["df"].columns = ["target", "ids", "date", "flag", "user", "text"]
        dataset["df"] = dataset["df"][["text", "target"]]
        dataset["df"].columns = ["sentence", "label"]
        dataset["df"] = process_convert_to_positive_vs_rest_annotation(dataset["df"])
        dataset["df"] = cls.__process_shuffle_rows(dataset["df"])
        dataset["df"] = cls.__process_annotate_source(
            dataset["df"], cls.__sources["twitter"]
        )
        cls.__df_describe(dataset["df"], dataset["name"])
        return dataset

    @classmethod
    def preprocess_all_datasets(cls):
        logger.info(f"Datasets directory: {cls.__datasets_directory}")
        complete_dataset = pd.DataFrame({"sentence": [], "label": [], "source": []})
        _16_milion_tweets = cls.process_16_million_tweets()
        complete_dataset = pd.merge(
            complete_dataset, _16_milion_tweets["df"], how="outer"
        )
        _186800_tweets = cls.process_186800_tweets()
        complete_dataset = pd.merge(complete_dataset, _186800_tweets["df"], how="outer")
        _36800_reddit_dataset = cls.process_36800_reddit_posts()
        complete_dataset = pd.merge(
            complete_dataset, _36800_reddit_dataset["df"], how="outer"
        )

        complete_dataset = cls.__process_shuffle_rows(complete_dataset)
        train_split_fraction = 0.7
        row_at_split = int(complete_dataset.shape[0] * train_split_fraction)
        train_set, test_set = (
            complete_dataset[:row_at_split],
            complete_dataset[row_at_split:],
        )

        joblib.dump(train_set, cls.__datasets_directory / "twitter_reddit_train.joblib")
        joblib.dump(test_set, cls.__datasets_directory / "twitter_reddit_test.joblib")
        train_set.to_csv(cls.__datasets_directory / "twitter_reddit_train.csv")
        test_set.to_csv(cls.__datasets_directory / "twitter_reddit_test.csv")

        cls.__df_describe(complete_dataset, "complete dataset")

    @classmethod
    def preprocess_training_data(cls):
        detected_num_cores = numexpr.detect_number_of_cores()
        logger.info(f"Number of cores detected on this machine: {detected_num_cores}")
        os.environ["NUMEXPR_MAX_THREADS"] = str(detected_num_cores)
        cls.preprocess_all_datasets()


if __name__ == "__main__":
    SentimentalAnalysisDataPreprocessor.preprocess_training_data()
