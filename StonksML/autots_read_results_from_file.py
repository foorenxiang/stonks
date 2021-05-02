from joblib import load
import os
from pathlib import Path
from utils import paths_catalog
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AutoTSData:
    CURRENT_DIRECTORY = Path(__file__).resolve().parent
    __DEFAULT_MODEL_DUMPS_FOLDER = paths_catalog.AUTOTS_MODEL_DUMPS

    FORECASTS_OPEN = "*forecasts_open.joblib"
    MODEL_RESULTS_OPEN = "*model_results_open.joblib"
    PREDICTION_RESULTS_OPEN = "*prediction_open.joblib"
    VALIDATION_RESULTS_OPEN = "*validation_results_open.joblib"
    FORECASTS_CLOSE = "*forecasts_close.joblib"
    MODEL_RESULTS_CLOSE = "*model_results_close.joblib"
    PREDICTION_RESULTS_CLOSE = "*prediction_close.joblib"
    VALIDATION_RESULTS_CLOSE = "*validation_results_close.joblib"
    FORECASTS_WIDE_DATASET = "*forecasts_wide_dataset.joblib"

    @staticmethod
    def __df_to_csv(file_data, model_dumps, stock_name, data_of_interest):
        try:
            file_data.to_csv(
                path_or_buf=(model_dumps / f"{stock_name}_{data_of_interest}.csv")
            )
        except AttributeError:
            logger.warning(f"{file_data} cannot be converted to csv, skipping...")

    @classmethod
    def print_autots_data(cls, data_of_interest, dump_folder=""):
        data_of_interest_name = data_of_interest[1:].split(".")[0]

        model_dumps = Path(dump_folder)
        if dump_folder == "":

            model_dumps_folder = cls.__DEFAULT_MODEL_DUMPS_FOLDER
            model_dumps_subdirectories = model_dumps_folder.glob("*")

            all_subdirs = [
                directory
                for directory in model_dumps_subdirectories
                if directory.is_dir()
            ]
            latest_subdir = max(all_subdirs, key=os.path.getmtime)

            dump_folder = latest_subdir
            model_dumps = dump_folder

        else:
            model_dumps = Path(dump_folder)

        try:
            files = model_dumps.glob(data_of_interest)
        except Exception:
            logger.error(f"Failed to glob {data_of_interest} in {dump_folder}")
            raise

        for file in files:
            stock_name = str(file.stem).split("_")[0]
            file_data = load(file)
            if (
                data_of_interest == cls.VALIDATION_RESULTS_OPEN
                or data_of_interest == cls.VALIDATION_RESULTS_CLOSE
            ):
                cls.__df_to_csv(
                    file_data, model_dumps, stock_name, data_of_interest_name
                )
            elif (
                data_of_interest == cls.FORECASTS_OPEN
                or data_of_interest == cls.FORECASTS_CLOSE
            ):
                cls.__df_to_csv(
                    file_data, model_dumps, stock_name, data_of_interest_name
                )
            elif data_of_interest == cls.FORECASTS_WIDE_DATASET:
                cls.__df_to_csv(
                    file_data, model_dumps, stock_name, data_of_interest_name
                )

            print("\n")
            print(stock_name)
            print(file_data)


def generate_results():
    AutoTSData.print_autots_data(AutoTSData.FORECASTS_OPEN)
    AutoTSData.print_autots_data(AutoTSData.FORECASTS_CLOSE)
    AutoTSData.print_autots_data(AutoTSData.FORECASTS_WIDE_DATASET)


if __name__ == "__main__":
    generate_results()
