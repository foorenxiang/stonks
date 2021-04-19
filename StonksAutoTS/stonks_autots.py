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


def get_stocks_data():
    stocks = get_stocks()
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


def dump_path(save_location, ticker_name, name, predictionTarget):
    return (
        f"{save_location}/{f'{ticker_name}_{name}_{predictionTarget.lower()}'}.joblib"
    )


def train_model(selectedMode, save_location, ticker_name, ticker_dfs, predictionTarget):
    model = AutoTSConfigs.create_model_lambda(selectedMode)().fit(
        ticker_dfs,
        date_col="DateCol",
        value_col=predictionTarget,
    )

    prediction = model.predict()
    # Print the details of the best model
    print("Details of best model:")
    print(model)

    # point forecasts dataframe
    forecasts = prediction.forecast
    # upper and lower forecasts
    forecasts_up, forecasts_low = (
        prediction.upper_forecast,
        prediction.lower_forecast,
    )

    # accuracy of all tried model results
    model_results = model.results()
    # and aggregated from cross validation
    validation_results = model.results("validation")

    variables_to_be_saved = {
        "model": model,
        "prediction": prediction,
        "forecasts": forecasts,
        "model_results": model_results,
        "validation_results": validation_results,
    }

    [
        dump(var, dump_path(save_location, ticker_name, name, predictionTarget))
        for name, var in variables_to_be_saved.items()
    ]


def create_model_dump_directories(selectedMode):
    dump_directory = f"{str(datetime.now()).split('.')[0]} {selectedMode}"
    dump_directory_parent = Path(__file__).resolve().parent
    (dump_directory_parent / "model_dumps").mkdir(exist_ok=True)
    (dump_directory_parent / "model_dumps" / dump_directory).mkdir(exist_ok=True)
    absolute_dump_directory = dump_directory_parent / "model_dumps" / dump_directory
    return absolute_dump_directory


def train_stonks():
    ticker_dfs = get_stocks_data()

    selectedMode = AutoTSConfigs.FAST
    print(f"Training with {selectedMode} mode")

    absolute_dump_directory = create_model_dump_directories(selectedMode)
    for ticker_name, ticker_dfs in ticker_dfs.items():
        ticker_dfs = set_date_on_yf_df(ticker_dfs)
        train_model(
            selectedMode, absolute_dump_directory, ticker_name, ticker_dfs, "Open"
        )
        train_model(
            selectedMode, absolute_dump_directory, ticker_name, ticker_dfs, "Close"
        )


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