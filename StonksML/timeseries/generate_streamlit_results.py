from joblib import load
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

import sys
from from_root import from_root

sys.path.append(str(from_root(".")))
from utils import paths_catalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitResults:
    CURRENT_DIRECTORY = Path(__file__).resolve().parent
    __DEFAULT_MODEL_DUMPS_PARENT_FOLDER = paths_catalog.AUTOTS_MODEL_DUMPS

    FORECASTS_OPEN = "*forecasts_open.joblib"
    FORECASTS_CLOSE = "*forecasts_close.joblib"
    FORECASTS_WIDE_DATASET = "*forecasts_wide_dataset.joblib"

    @staticmethod
    def __get_models_date():
        pass

    @classmethod
    def __fetch_models(cls, dump_folder):
        if dump_folder and type(dump_folder) == str:
            model_dumps = Path(dump_folder)
        else:
            MODEL_DUMPS_PARENT_FOLDER = cls.__DEFAULT_MODEL_DUMPS_PARENT_FOLDER
            model_dumps_subdirectories = MODEL_DUMPS_PARENT_FOLDER.glob("*")

            all_subdirs = [
                directory
                for directory in model_dumps_subdirectories
                if directory.is_dir()
            ]
            last_modified_subdirectory = max(all_subdirs, key=os.path.getmtime)
            model_dumps = last_modified_subdirectory

        model_trained_timestamp_string = " ".join(str(model_dumps.stem).split(" ")[0:2])
        model_trained_timestamp = datetime.strptime(
            model_trained_timestamp_string, "%Y-%m-%d %H:%M:%S"
        )

        time_since_model_training = datetime.now() - model_trained_timestamp
        return time_since_model_training, model_dumps

    @classmethod
    def retrieve_autots_forecasts(cls, data_of_interest, dump_folder=None):
        data_of_interest_name = data_of_interest[1:].split(".")[0]
        logger.warning("Check __fetch_models functionality")

        time_since_model_training, model_dumps = cls.__fetch_models(dump_folder)

        logger.info(f"Existing models were trained {time_since_model_training} ago")

        is_trained_within_half_day = time_since_model_training < timedelta(hours=12)

        if not is_trained_within_half_day:
            logger.warning(f"Models are outdated, retraining on latest data")
            from autots_train_and_predict import train_stonks

            train_stonks()
            logger.info(f"Models finished training on latest data")
            time_since_model_training, model_dumps = cls.__fetch_models(dump_folder)

        try:
            files = model_dumps.glob(data_of_interest)
        except Exception:
            logger.error(f"Failed to glob {data_of_interest} in {dump_folder}")
            raise

        for file in files:
            stock_name = str(file.stem).split("_")[0]
            file_data = load(file)
            # if (
            #     data_of_interest == cls.FORECASTS_OPEN
            #     or data_of_interest == cls.FORECASTS_CLOSE
            # ):
            #     cls.__df_to_csv(
            #         file_data, model_dumps, stock_name, data_of_interest_name
            #     )
            # elif data_of_interest == cls.FORECASTS_WIDE_DATASET:
            #     cls.__df_to_csv(
            #         file_data, model_dumps, stock_name, data_of_interest_name
            #     )

            print("\n")
            print(stock_name)
            print(file_data)


def generate_streamlit_results():
    StreamlitResults.retrieve_autots_forecasts(StreamlitResults.FORECASTS_OPEN)
    StreamlitResults.retrieve_autots_forecasts(StreamlitResults.FORECASTS_CLOSE)
    StreamlitResults.retrieve_autots_forecasts(StreamlitResults.FORECASTS_WIDE_DATASET)


if __name__ == "__main__":
    logger.info(generate_streamlit_results())
