from datetime import datetime, date
import logging
import yfinance as yf
from joblib import dump
from autots import AutoTS
from pathlib import Path
from datetime import datetime, timedelta

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
        stock: yf.Ticker(stock).history(period=period, end=get_last_weekday())
        for stock in stocks
    }
    return stocks_dfs_dict


def get_last_weekday():
    last_weekday = datetime.today()

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

    __autots_mode_configs = {
        "FAST": lambda: AutoTS(
            forecast_length=14,
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
            forecast_length=14,
            frequency="infer",
            prediction_interval=0.95,
            ensemble="all",
            model_list="parallel",
            transformer_list="fast",
            max_generations=15,
            num_validations=2,
            validation_method="backwards",
        ),
    }

    @classmethod
    def create_model_lambda(cls, selectedMode):
        return cls.__autots_mode_configs[getattr(cls, selectedMode)]


def train_stonks():

    stocks = get_stocks()
    ticker_dfs = get_stocks_data(stocks)

    print(ticker_dfs["AAPL"])

    for key, item in ticker_dfs.items():
        print(key)
        print(item)
        print("\n\n\n")

    selectedMode = AutoTSConfigs.ACCURATE

    target_name = f"{str(datetime.now()).split('.')[0]} {selectedMode}"
    target_parent = Path(__file__).resolve().parent
    (target_parent / "model_dumps").mkdir(exist_ok=True)
    (target_parent / "model_dumps" / target_name).mkdir(exist_ok=True)
    target = target_parent / "model_dumps" / target_name

    for ticker_name, ticker_dfs in ticker_dfs.items():

        ticker_dfs = set_date_on_yf_df(ticker_dfs)

        model = AutoTSConfigs.create_model_lambda(selectedMode)()
        model = model.fit(
            ticker_dfs,
            date_col="DateCol",
            value_col="Close",
        )

        prediction = model.predict()
        # Print the details of the best model
        print("Details of best model:")
        print(model)

        # point forecasts dataframe
        forecasts_df = prediction.forecast
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
            "forecasts_df": forecasts_df,
            # "forecasts_up": forecasts_up,
            # "forecasts_low": forecasts_low,
            "model_results": model_results,
            "validation_results": validation_results,
        }

        [
            dump(var, f"{target}/{f'{ticker_name}_{name}'}.joblib")
            for name, var in variables_to_be_saved.items()
        ]


if __name__ == "__main__":
    # time_exec(lambda: train_stonks(forecast_days=7))

    # train_stonks()
    import time

    start_time = time.monotonic()
    train_stonks()
    end_time = time.monotonic()
    print(f"Time taken to train: {end_time - start_time}")