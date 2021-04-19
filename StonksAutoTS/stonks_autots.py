import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import time


def time_exec(lambdaFunc):

    start_time = time.monotonic()
    lambdaFunc()
    end_time = time.monotonic()
    print(f"Time taken to train: {end_time - start_time}")


import yfinance as yf
from joblib import dump
from pathlib import Path
from datetime import datetime, timedelta
from AutoTSConfigs import AutoTSConfigs


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


def train_model_close(selectedMode, target, ticker_name, ticker_dfs):
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

    variables_to_be_saved = {
        "model_close": model_close,
        "prediction_close": prediction_close,
        "forecasts_close": forecasts_close,
        "model_results_close": model_results_close,
        "validation_results_close": validation_results_close,
    }

    [
        dump(var, f"{target}/{f'{ticker_name}_{name}'}.joblib")
        for name, var in variables_to_be_saved.items()
    ]


def train_model_open(selectedMode, target, ticker_name, ticker_dfs):
    model_open = AutoTSConfigs.create_model_lambda(selectedMode)().fit(
        ticker_dfs,
        date_col="DateCol",
        value_col="Open",
    )

    prediction_open = model_open.predict()
    # Print the details of the best model
    print("Details of best model:")
    print(model_open)

    # point forecasts dataframe
    forecasts_open = prediction_open.forecast
    # upper and lower forecasts
    forecasts_up, forecasts_low = (
        prediction_open.upper_forecast,
        prediction_open.lower_forecast,
    )

    # accuracy of all tried model results
    model_results_open = model_open.results()
    # and aggregated from cross validation
    validation_results_open = model_open.results("validation")

    variables_to_be_saved = {
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


# TODO: refactor to make this function smaller (consider using class)
def train_stonks():

    stocks = get_stocks()
    ticker_dfs = get_stocks_data(stocks)

    selectedMode = AutoTSConfigs.FAST
    print(f"Training with {selectedMode} mode")

    target_name = f"{str(datetime.now()).split('.')[0]} {selectedMode}"
    target_parent = Path(__file__).resolve().parent
    (target_parent / "model_dumps").mkdir(exist_ok=True)
    (target_parent / "model_dumps" / target_name).mkdir(exist_ok=True)
    target = target_parent / "model_dumps" / target_name

    for ticker_name, ticker_dfs in ticker_dfs.items():

        ticker_dfs = set_date_on_yf_df(ticker_dfs)

        train_model_open(selectedMode, target, ticker_name, ticker_dfs)
        train_model_close(selectedMode, target, ticker_name, ticker_dfs)


if __name__ == "__main__":
    import math
    import time

    start_time = time.monotonic()
    train_stonks()
    end_time = time.monotonic()
    time_taken = end_time - start_time
    mins, secs = math.floor(time_taken / 60), math.floor(time_taken % 60)
    print(f"Time taken to train: {mins}mins {secs}secs")
    from cli_read_results import AutoTSData

    AutoTSData.print_autots_data(AutoTSData.FORECASTS_OPEN)
    AutoTSData.print_autots_data(AutoTSData.FORECASTS_CLOSE)