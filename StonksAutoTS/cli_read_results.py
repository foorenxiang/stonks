from joblib import load
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AutoTSData:

    FORECASTS = "*forecasts_df*.joblib"
    MODEL_RESULTS = "*model_results.joblib"
    PREDICTION_RESULTS = "*prediction.joblib"
    VALIDATION_RESULTS = "*validation_results.joblib"

    @staticmethod
    def __df_to_csv(file_data, model_dumps, stock_name, data_of_interest):
        file_data.to_csv(
            path_or_buf=(model_dumps / f"{stock_name}_{data_of_interest}.csv")
        )

    @classmethod
    def print_autots_data(cls, data_of_interest, dump_folder):
        data_of_interest_name = data_of_interest[1:].split(".")[0].split("*")[0]

        model_dumps = Path(dump_folder)
        # if dump_folder == None:
        # dump_folder = (Path(__file__).resolve().parent / "model_dumps").glob()

        # else:
        # model_dumps = Path(dump_folder)

        try:
            files = model_dumps.glob(data_of_interest)
        except Exception:
            logger.error(f"Failed to glob {data_of_interest} in {dump_folder}")
            raise

        for file in files:
            stock_name = str(file.stem).split("_")[0]
            file_data = load(file)
            if data_of_interest == cls.VALIDATION_RESULTS:
                cls.__df_to_csv(
                    file_data, model_dumps, stock_name, data_of_interest_name
                )
            elif data_of_interest == cls.FORECASTS:
                cls.__df_to_csv(
                    file_data,
                    model_dumps,
                    stock_name,
                    data_of_interest_name.split("_")[0],
                )
            print("\n")
            print(stock_name)
            print(file_data)


if __name__ == "__main__":
    AutoTSData.print_autots_data(
        AutoTSData.FORECASTS,
        r"/Users/foorx/code/stonks/StonksAutoTS/model_dumps/2021-04-18 15:34:23.007267",
    )
