from datetime import datetime, date
import logging
import yfinance as yf
from joblib import dump
from autots import AutoTS
from pathlib import Path
from datetime import datetime, timedelta
import math
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def time_exec(lambdaFunc):
    import time

    start_time = time.monotonic()
    lambdaFunc()
    end_time = time.monotonic()
    print(f"Time taken to train: {end_time - start_time}")


def get_stocks():
    from ticker_symbols import get_ticker_symbols

    return get_ticker_symbols()


def get_stocks_data(stocks):
    period = "3mo"
    stocks_dfs_dict = {
        stock: yf.Ticker(stock).history(
            period=period, end=get_last_weekday(datetime.today())
        )
        for stock in stocks
    }
    return stocks_dfs_dict


def get_last_weekday(date):
    last_weekday = date

    while last_weekday.weekday() > 4:
        last_weekday -= timedelta(days=1)

    return str(last_weekday.date())


def set_date_on_yf_df(yf_df):
    """Might have performance issue"""
    yf_df["DateCol"] = yf_df.apply(lambda row: row.name, axis=1)
    return yf_df


class AutoTSConfigs:
    FAST = "FAST"
    ACCURATE = "ACCURATE"
    FORECAST_LENGTH = 7

    __suggested_validation_trials = math.floor(FORECAST_LENGTH / 3)
    validation_trials = (
        __suggested_validation_trials if __suggested_validation_trials > 2 else 2
    )

    __autots_mode_configs = {
        "FAST": lambda: AutoTS(
            forecast_length=AutoTSConfigs.FORECAST_LENGTH,
            frequency="infer",
            prediction_interval=0.9,
            ensemble=None,
            model_list="superfast",
            transformer_list="fast",
            max_generations=5,
            num_validations=2,
            validation_method="backwards",
        ),
        "ACCURATE": lambda: AutoTS(
            forecast_length=AutoTSConfigs.FORECAST_LENGTH,
            frequency="infer",
            prediction_interval=0.9,
            ensemble="dist",
            model_list="parallel",
            transformer_list="fast",
            max_generations=5,
            num_validations=AutoTSConfigs.validation_trials,
            validation_method="backwards",
        ),
    }

    @classmethod
    def create_model_lambda(cls, selectedMode):
        return cls.__autots_mode_configs[getattr(cls, selectedMode)]


# TODO: refactor to make this function smaller (consider using class)
def train_stonks():

    stocks = get_stocks()
    ticker_dfs = get_stocks_data(stocks)

    selectedMode = AutoTSConfigs.FAST

    target_name = f"{str(datetime.now()).split('.')[0]} {selectedMode}"
    target_parent = Path(__file__).resolve().parent
    (target_parent / "model_dumps").mkdir(exist_ok=True)
    (target_parent / "model_dumps" / target_name).mkdir(exist_ok=True)
    target = target_parent / "model_dumps" / target_name

    for ticker_name, ticker_dfs in ticker_dfs.items():

        ticker_dfs = set_date_on_yf_df(ticker_dfs)

        model_close = AutoTSConfigs.create_model_lambda(selectedMode)().fit(
            ticker_dfs,
            date_col="DateCol",
            value_col="Close",
        )

        prediction_close = model_close.predict()
        # Print the details of the best model
        print("Details of best model:")
        print(model_close)

        # point forecasts dataframe
        forecasts_close = prediction_close.forecast
        # upper and lower forecasts
        forecasts_up, forecasts_low = (
            prediction_close.upper_forecast,
            prediction_close.lower_forecast,
        )

        # accuracy of all tried model results
        model_results_close = model_close.results()
        # and aggregated from cross validation
        validation_results_close = model_close.results("validation")

        model_open = AutoTSConfigs.create_model_lambda(selectedMode)().fit(
            ticker_dfs,
            date_col="DateCol",
            value_col="Open",
        )

        prediction_open = model_close.predict()
        # Print the details of the best model
        print("Details of best model:")
        print(model_close)

        # point forecasts dataframe
        forecasts_open = prediction_open.forecast
        # upper and lower forecasts
        forecasts_up, forecasts_low = (
            prediction_open.upper_forecast,
            prediction_open.lower_forecast,
        )

        # accuracy of all tried model results
        model_results_open = model_close.results()
        # and aggregated from cross validation
        validation_results_open = model_close.results("validation")

        variables_to_be_saved = {
            "model_close": model_close,
            "prediction_close": prediction_close,
            "forecasts_close": forecasts_close,
            "model_results_close": model_results_close,
            "validation_results_close": validation_results_close,
            "model_open": model_open,
            "prediction_open": prediction_open,
            "forecasts_open": forecasts_open,
            "model_results_open": model_results_open,
            "validation_results_open": validation_results_open,
        }

        [
            dump(var, f"{target}/{f'{ticker_name}_{name}'}.joblib")
            for name, var in variables_to_be_saved.items()
        ]


if __name__ == "__main__":
    start_time = time.monotonic()
    train_stonks()
    end_time = time.monotonic()
    time_taken = end_time - start_time
    mins, secs = math.floor(time_taken / 60), math.floor(time_taken % 60)
    print(f"Time taken to train: {mins}mins {secs}secs")