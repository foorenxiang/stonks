# modified from https://towardsdatascience.com/train-multiple-time-series-forecasting-models-in-one-line-of-python-code-615f2253b67a
# modified from https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75

import yfinance as yf
from joblib import dump
from pathlib import Path
from datetime import datetime, timedelta
from ticker_symbols import get_ticker_symbols
import numexpr
import os
from autots_config import AutoTSConfigs
from utils.exec_time import exec_time
import logging

file_dir = Path(__file__).resolve().parent
log_path = file_dir / "autots_logs"
try:
    log_path.mkdir()
except FileExistsError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(filename=log_path / "autots_training.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__file__)


class StonksAutoTS:
    selectedMode = None
    __forecasts_generated_by_training = []
    __dataset_is_long = False  # True if only estimating based on pure time series, false if estimating based on other features in timeseries (wide dataset)

    @classmethod
    def __get_stocks_data(cls):
        stocks = get_ticker_symbols()
        period = "3mo"
        stocks_dfs_dict = {
            stock: yf.Ticker(stock).history(
                period=period, end=cls.__get_last_weekday(datetime.today())
            )
            for stock in stocks
        }
        return stocks_dfs_dict

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

        artifacts = {
            "model": model,
            "prediction": prediction,
            "forecasts": forecast,
            "model_results": model_results,
            "validation_results": validation_results,
        }

        [
            dump(
                artifact,
                f"""{save_location}/{f"{ticker_name}_{artifact_name}_{predictionTarget.lower() if predictionTarget else 'wide_dataset'}"}.joblib""",
            )
            for artifact_name, artifact in artifacts.items()
        ]

        cls.__forecasts_generated_by_training.append(
            {"ticker_name": ticker_name, "forecast": forecast}
        )

    @classmethod
    def __create_model_dump_directories(cls):
        dump_directory = f"{str(datetime.now()).split('.')[0]} {cls.selectedMode}"
        dump_directory_parent = Path(__file__).resolve().parent
        (dump_directory_parent / "model_dumps").mkdir(exist_ok=True)
        (dump_directory_parent / "model_dumps" / dump_directory).mkdir(exist_ok=True)
        absolute_dump_directory = dump_directory_parent / "model_dumps" / dump_directory
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
        ticker_dfs = cls.__get_stocks_data()

        cls.selectedMode = AutoTSConfigs.DEFAULT
        print(f"Training with {cls.selectedMode} mode")

        detected_num_cores = numexpr.detect_number_of_cores()
        logger.info(f"Number of cores detected on this machine: {detected_num_cores}")
        os.environ["NUMEXPR_MAX_THREADS"] = str(detected_num_cores)

        absolute_dump_directory = cls.__create_model_dump_directories()
        for ticker_name, ticker_dfs in ticker_dfs.items():
            ticker_dfs = cls.__set_date_on_yf_df(ticker_dfs)
            if cls.__dataset_is_long:
                cls.__train_model(
                    absolute_dump_directory, ticker_name, ticker_dfs, "Open"
                )
                cls.__train_model(
                    absolute_dump_directory, ticker_name, ticker_dfs, "Close"
                )
            else:
                cls.__train_model(absolute_dump_directory, ticker_name, ticker_dfs)


def train_stonks():
    StonksAutoTS.train_and_forecast_stonks()
    logger.info("\n\n\n Forecasts generated from latest data:")
    StonksAutoTS.log_results_from_training()


if __name__ == "__main__":
    train_stonks()
