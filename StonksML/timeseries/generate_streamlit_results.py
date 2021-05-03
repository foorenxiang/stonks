from joblib import load
import os
from pathlib import Path
from datetime import datetime, timedelta

import logging

import sys
from from_root import from_root

sys.path.append(str(from_root(".")))
from utils import paths_catalog
from utils.fire_and_forget_subprocess import fire_and_forget
from StonksML.timeseries.autots_train_and_predict import train_stonks

train_stonks = fire_and_forget(train_stonks)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitResults:
    CURRENT_DIRECTORY = Path(__file__).resolve().parent
    __DEFAULT_MODEL_DUMPS_PARENT_FOLDER = paths_catalog.AUTOTS_MODEL_DUMPS

    FORECASTS_OPEN = "*forecasts_open.joblib"
    FORECASTS_CLOSE = "*forecasts_close.joblib"
    FORECASTS_WIDE_DATASET = "*forecasts_wide_dataset.joblib"

    __stocks_data = []

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
        return model_dumps, time_since_model_training

    @classmethod
    def __check_if_need_retrain(cls, dump_folder, time_since_model_training):
        logger.info(f"Existing models were trained {time_since_model_training} ago")
        is_trained_within_half_day = time_since_model_training < timedelta(hours=12)

        if is_trained_within_half_day:
            logger.info("Using existing models")
            return dump_folder, time_since_model_training

        logger.warning("Models are outdated")
        train_stonks()
        logger.warning("Retraining on latest data")
        return None

    @classmethod
    def __process_autots_forecasts(cls, prediction_target, dump_folder=None):
        prediction_target_name = (
            prediction_target[1:].split(".")[0].split("forecasts_")[1]
        )
        logger.warning("Check __fetch_models functionality")

        check_result = cls.__check_if_need_retrain(*cls.__fetch_models(dump_folder))

        if type(check_result) != tuple:
            return None

        model_dumps = check_result[0]
        print(model_dumps)

        try:
            files = model_dumps.glob(prediction_target)
        except Exception:
            logger.error(f"Failed to glob {prediction_target} in {dump_folder}")
            raise

        for file in files:
            stock = str(file.stem).split("_")[0]
            prediction_dataframe = load(file)

            artifact = {"name": stock}
            artifact[prediction_target_name] = prediction_dataframe

            updated = False
            for idx, stock_data in enumerate(cls.__stocks_data):
                if stock_data["name"] == stock:
                    cls.__stocks_data[idx] = {**stock_data, **artifact}
                    updated = True
                    break

            if not updated:
                cls.__stocks_data.append(artifact)

    @classmethod
    def retrieve_autots_forecasts(cls):
        [
            cls.__process_autots_forecasts(prediction_target)
            for prediction_target in [
                StreamlitResults.FORECASTS_WIDE_DATASET,
                StreamlitResults.FORECASTS_OPEN,
                StreamlitResults.FORECASTS_CLOSE,
            ]
        ]

        return cls.__stocks_data


def generate_streamlit_results():
    stocks_data = StreamlitResults.retrieve_autots_forecasts()
    return stocks_data


if __name__ == "__main__":
    logger.info(generate_streamlit_results())
