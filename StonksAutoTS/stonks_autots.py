from datetime import datetime
import logging
import yfinance as yf
from joblib import dump
from autots import AutoTS
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_stocks():
    from StonksAutoTS.ticker_symbols import get_ticker_symbols

    return get_ticker_symbols()


def train_stonks():

    stocks = get_stocks()
    period = "max"

    ticker_dfs = {stock: yf.Ticker(stock).history(period=period) for stock in stocks}

    print(ticker_dfs)

    for key, item in ticker_dfs.items():
        print(key)
        print(item)
        print("\n\n\n")

    def set_date_on_yf_df(yf_df):
        """Might have performance issue"""
        yf_df["DateCol"] = yf_df.apply(lambda row: row.name, axis=1)
        return yf_df

    target_name = str(datetime.now())
    target_parent = Path(__file__)
    (target_parent / "model_dumps").mkdir(exist_ok=True)
    (target_parent / "model_dumps" / target_name).mkdir(exist_ok=True)
    target = target_parent / "model_dumps" / target_name

    for ticker_name, ticker_dfs in ticker_dfs.items():

        ticker_dfs = set_date_on_yf_df(ticker_dfs)

        model = AutoTS(
            forecast_length=3,
            frequency="infer",
            prediction_interval=0.9,
            ensemble=None,
            model_list="superfast",
            transformer_list="fast",
            max_generations=5,
            num_validations=2,
            validation_method="backwards",
        )
        model = model.fit(
            ticker_dfs,
            date_col="DateCol",
            value_col="Close",
        )

        prediction = model.predict()
        # Print the details of the best model
        print("Details of best model:")
        print(model)
        dump(model, "best_model.joblib")
        dump(prediction, "prediction.joblib")

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
            "forecasts_up": forecasts_up,
            "forecasts_low": forecasts_low,
            "model_results": model_results,
            "validation_results": validation_results,
        }

        [
            dump(var, f"{target}/{f'{ticker_name}_{name}'}.joblib")
            for name, var in variables_to_be_saved.items()
        ]


if __name__ == "__main__":
    train_stonks()