# modified from https://towardsdatascience.com/train-multiple-time-series-forecasting-models-in-one-line-of-python-code-615f2253b67a
# modified from https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75

from mlflow.tracking.fluent import log_artifacts, log_params
import yfinance as yf
from joblib import dump
from pathlib import Path
from datetime import datetime, timedelta
import numexpr
import os
import logging
from mlflow import pyfunc, log_param, log_params, log_artifact
from mlflow.exceptions import MlflowException
from autots_config import AutoTSConfigs
from shutil import rmtree


import sys
from from_root import from_root

sys.path.append(str(from_root(".")))
from utils import paths_catalog
from utils.exec_time import exec_time
from utils.ticker_symbols import get_ticker_symbols
from utils.get_dataset_catalog import get_dataset_catalog
from utils.mlflow_wrapper import GenericModel


CURRENT_DIRECTORY = Path(__file__).resolve().parent
LOG_PATH = paths_catalog.AUTOTS_LOGS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler())

try:
    LOG_PATH.mkdir()
except FileExistsError:
    logger.info(f"{LOG_PATH} exists, using it")

logger.addHandler(
    logging.FileHandler(filename=LOG_PATH / "autots_training.log", mode="w")
)


class StonksAutoTS:
    selectedMode = AutoTSConfigs.FAST
    __forecasts_generated_by_training = []
    __dataset_is_long = False  # True if only estimating based on pure time series, false if estimating based on other features in timeseries (wide dataset)
    __stocks_data_dump_location = from_root(
        get_dataset_catalog()["datasets"]["generated"]["Stocks Data"]["location"]
    )

    @classmethod
    def __get_stocks_data(cls):
        stocks = get_ticker_symbols()
        period = "3mo"
        end_date = cls.__get_last_weekday(datetime.today())
        stocks_dfs_dict = {
            stock: yf.Ticker(stock).history(period=period, end=end_date)
            for stock in stocks
        }
        log_params({"stocks": stocks, "stocks_period": period, "end_date": end_date})

        return stocks_dfs_dict, end_date

    @staticmethod
    def __get_last_weekday(date):
        last_weekday = date
        while last_weekday.weekday() > 4:
            last_weekday -= timedelta(days=1)
        return str(last_weekday.date())

    @staticmethod
    def __set_date_on_yf_df(yf_df):
        """Might have performance issue"""
        yf_df["DateCol"] = yf_df.apply(lambda row: row.name, axis=1)
        return yf_df

    @staticmethod
    def __purge_existing_identical_mlflow_model(absolute_mlflow_model_path):
        if Path(absolute_mlflow_model_path).exists():
            rmtree(absolute_mlflow_model_path)

    @classmethod
    def __mlflow_save_model(cls, ticker_name, predictionTarget, model):
        mlflow_relative_model_path = f"{ticker_name}_{predictionTarget if predictionTarget else 'wide'}_{cls.__get_last_weekday(datetime.today())}_model"

        absolute_mlflow_model_path = str(
            paths_catalog.AUTOTS_MLFLOW_MODEL_DUMP / mlflow_relative_model_path
        )

        cls.__purge_existing_identical_mlflow_model(absolute_mlflow_model_path)

        mlflow_python_model = GenericModel(model)

        pyfunc.save_model(
            path=absolute_mlflow_model_path, python_model=mlflow_python_model
        )

        pyfunc.log_model(
            artifact_path=mlflow_relative_model_path, python_model=mlflow_python_model
        )

        # loaded_model = pyfunc.load_model(absolute_mlflow_model_path)

        # prediction = loaded_model.predict(None)
        # forecast = prediction.forecast

    @classmethod
    def __train_model(
        cls, save_location, ticker_name, ticker_dfs, predictionTarget=None
    ):
        if predictionTarget:
            logger.info(
                f"Training model for {ticker_name} with {predictionTarget} prediction target"
            )
        else:
            logger.info(f"Training model for {ticker_name} with wide features")

        model = AutoTSConfigs.create_model_lambda(cls.selectedMode)().fit(
            ticker_dfs,
            date_col="DateCol" if predictionTarget else None,
            value_col=predictionTarget,
        )

        prediction = model.predict()
        forecast = prediction.forecast
        model_results = model.results()
        validation_results = model.results("validation")

        cls.__mlflow_save_model(ticker_name, predictionTarget, model)

        artifacts = {
            "model": model,
            "prediction": prediction,
            "forecasts": forecast,
            "model_results": model_results,
            "validation_results": validation_results,
        }

        for artifact_name, artifact in artifacts.items():
            dumpfile_path = f"""{save_location}/{f"{ticker_name}_{artifact_name}_{predictionTarget.lower() if predictionTarget else 'wide_dataset'}"}.joblib"""
            dump(
                artifact,
                dumpfile_path,
            )
            log_artifact(dumpfile_path)

        cls.__forecasts_generated_by_training.append(
            {"ticker_name": ticker_name, "forecast": forecast}
        )

    @classmethod
    def __create_model_dump_directories(cls):
        dump_directory = f"{str(datetime.now()).split('.')[0]} {cls.selectedMode}"
        dump_directory_parent = paths_catalog.AUTOTS_MODEL_DUMPS
        (dump_directory_parent).mkdir(exist_ok=True)
        (dump_directory_parent / dump_directory).mkdir(exist_ok=True)
        absolute_dump_directory = dump_directory_parent / dump_directory
        return absolute_dump_directory

    @classmethod
    def log_results_from_training(cls):
        if cls.__forecasts_generated_by_training:
            for forecast in cls.__forecasts_generated_by_training:
                logger.info(f"Forecast for {forecast['ticker_name']}:")
                logger.info(forecast["forecast"])
            return
        logger.warning(
            "No forecasts present from model training! Have you trained any models yet?"
        )
        raise ValueError

    @classmethod
    @exec_time
    def train_and_forecast_stonks(cls):
        ticker_dfs, end_date = cls.__get_stocks_data()

        dump_location = (
            cls.__stocks_data_dump_location / f"{end_date}_stocks_data.joblib"
        )

        dump(ticker_dfs, dump_location)
        log_artifact(dump_location)

        print(f"Training AutoTS models with {cls.selectedMode} config")

        detected_num_cores = numexpr.detect_number_of_cores()
        logger.info(f"Number of cores detected on this machine: {detected_num_cores}")
        os.environ["NUMEXPR_MAX_THREADS"] = str(detected_num_cores)

        absolute_dump_directory = cls.__create_model_dump_directories()
        for ticker_name, ticker_dfs in ticker_dfs.items():
            ticker_dfs = cls.__set_date_on_yf_df(ticker_dfs)
            if cls.__dataset_is_long:
                prediction_targets = "Open", "Close"
                [
                    cls.__train_model(
                        absolute_dump_directory,
                        ticker_name,
                        ticker_dfs,
                        prediction_target,
                    )
                    for prediction_target in prediction_targets
                ]
            else:
                cls.__train_model(absolute_dump_directory, ticker_name, ticker_dfs)


def train_stonks():
    StonksAutoTS.train_and_forecast_stonks()
    logger.info("\n\n\n Forecasts generated from latest data:")
    StonksAutoTS.log_results_from_training()
    log_param("AutoTS mode", StonksAutoTS.selectedMode)


if __name__ == "__main__":
    train_stonks()
