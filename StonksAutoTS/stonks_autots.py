import yfinance as yf
from joblib import dump
from pathlib import Path
from datetime import datetime, timedelta
from ticker_symbols import get_ticker_symbols
from AutoTSConfigs import AutoTSConfigs
from utils.exec_time import exec_time


class StonksAutoTS:
    selectedMode = None

    @staticmethod
    def __get_stocks():
        return get_ticker_symbols()

    @classmethod
    def __get_stocks_data(cls):
        stocks = cls.__get_stocks()
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

    @staticmethod
    def __dump_path(save_location, ticker_name, name, predictionTarget):
        return f"{save_location}/{f'{ticker_name}_{name}_{predictionTarget.lower()}'}.joblib"

    @classmethod
    def __train_model(cls, save_location, ticker_name, ticker_dfs, predictionTarget):
        model = AutoTSConfigs.create_model_lambda(cls.selectedMode)().fit(
            ticker_dfs,
            date_col="DateCol",
            value_col=predictionTarget,
        )
        prediction = model.predict()
        forecasts = prediction.forecast
        model_results = model.results()
        validation_results = model.results("validation")

        variables_to_be_saved = {
            "model": model,
            "prediction": prediction,
            "forecasts": forecasts,
            "model_results": model_results,
            "validation_results": validation_results,
        }

        [
            dump(
                var, cls.__dump_path(save_location, ticker_name, name, predictionTarget)
            )
            for name, var in variables_to_be_saved.items()
        ]

    @classmethod
    def __create_model_dump_directories(cls):
        dump_directory = f"{str(datetime.now()).split('.')[0]} {cls.selectedMode}"
        dump_directory_parent = Path(__file__).resolve().parent
        (dump_directory_parent / "model_dumps").mkdir(exist_ok=True)
        (dump_directory_parent / "model_dumps" / dump_directory).mkdir(exist_ok=True)
        absolute_dump_directory = dump_directory_parent / "model_dumps" / dump_directory
        return absolute_dump_directory

    @classmethod
    @exec_time
    def train_stonks(cls):
        ticker_dfs = cls.__get_stocks_data()

        cls.selectedMode = AutoTSConfigs.FAST
        print(f"Training with {cls.selectedMode} mode")

        absolute_dump_directory = cls.__create_model_dump_directories()
        for ticker_name, ticker_dfs in ticker_dfs.items():
            ticker_dfs = cls.__set_date_on_yf_df(ticker_dfs)
            cls.__train_model(absolute_dump_directory, ticker_name, ticker_dfs, "Open")
            cls.__train_model(absolute_dump_directory, ticker_name, ticker_dfs, "Close")


if __name__ == "__main__":
    StonksAutoTS.train_stonks()
    from cli_read_results import generate_results

    generate_results()
