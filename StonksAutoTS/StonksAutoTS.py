import yfinance as yf
from joblib import dump


aapl = yf.Ticker("AAPL")
aapl_df = aapl.history(period="max")


def describe_stock(stock):
    from pprint import pformat

    print(pformat(aapl.info))


# describe_stock(aapl)


def set_date_on_yf_df(yf_df):
    """Might have performance issue"""
    yf_df["DateCol"] = yf_df.apply(lambda row: row.name, axis=1)
    return yf_df


aapl_df = set_date_on_yf_df(aapl_df)


from autots import AutoTS

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
    aapl_df,
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
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast


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

from pathlib import Path

target_name = "testdump"
target_parent = Path(__file__).resolve().parent
target = target_parent / target_name
(target_parent / target_name).mkdir(parents=True, exist_ok=True)
[dump(var, f"{target}/{name}.joblib") for name, var in variables_to_be_saved.items()]
