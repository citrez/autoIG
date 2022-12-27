from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import set_config
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from autoIG.config import Source
from autoIG.instruments import Epics, Tickers
from autoIG.modelling import (
    adapt_IG_data_for_training,
    adapt_YF_data_for_training,
    create_past_ask_Open,
    fillna_,
    normalise_,
)
from autoIG.utils import print_shape

EXPERIMENT_NAME = "stacked-linear-reg"
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(registered_model_name="stacked-linear-reg-model")
set_config(transform_output="pandas")

model_config = dict()
historical_prices_config = dict()

model_config["RELOAD_DATA"] = False
model_config["SOURCE"] = Source.yahoo_finance.value
model_config["NUMBER_OF_PAST_ASKS"] = 15  # This is for training.
model_config["EPIC"] = Epics.US_CRUDE_OIL.value
model_config["TICKER"] = Tickers.US_CRUDE_OIL.value

historical_prices_config["resolution"] = "1Min"
historical_prices_config["numpoints"] = 500

if model_config["RELOAD_DATA"]:
    if model_config["SOURCE"] == "IG":
        from trading_ig.rest import IGService

        from autoIG.config import ig_service_config

        ig_service = IGService(**ig_service_config)
        ig = ig_service.create_session()
        results_ = ig_service.fetch_historical_prices_by_epic(
            model_config["EPIC"], **historical_prices_config
        )
        model_data = results_["prices"]
        model_data.to_pickle("model_data_ig.pkl")
    if model_config["SOURCE"] == "YF":
        import yfinance as yf

        ticker = yf.Ticker(model_config["TICKER"])
        model_data = ticker.history(interval="1m", start="2022-12-05", end="2022-12-10")
        model_data.to_pickle("model_data_yf.pkl")
    else:
        Exception("Please provide source to reload data from: (IG/YF)")

else:
    if model_config["SOURCE"] == "IG":
        model_data = pd.read_pickle(Path(__file__).parent / "model_data_ig.pkl")
    if model_config["SOURCE"] == "YF":
        model_data = pd.read_pickle(Path(__file__).parent / "model_data_yf.pkl")

if model_config["SOURCE"] == "IG":
    model_data = model_data.pipe(adapt_IG_data_for_training)
if model_config["SOURCE"] == "YF":
    model_data = model_data.pipe(adapt_YF_data_for_training)
model_data.pipe(print_shape)

create_past_ask_Open_num_small = partial(create_past_ask_Open, num=3)
create_past_ask_Open_num_medium = partial(create_past_ask_Open, num=10)
create_past_ask_Open_num_large = partial(create_past_ask_Open, num=15)

fillna_transformer = FunctionTransformer(fillna_)
normalise_transformer = FunctionTransformer(normalise_)

pl1 = Pipeline(
    [
        (
            "add_past_period_columns",
            FunctionTransformer(create_past_ask_Open_num_small),
        ),
        ("fill_na", fillna_transformer),
        ("normalise", normalise_transformer),
        ("predictor", LinearRegression()),
    ]
)
pl2 = Pipeline(
    [
        (
            "add_past_period_columns",
            FunctionTransformer(create_past_ask_Open_num_medium),
        ),
        ("fill_na", fillna_transformer),
        ("normalise", normalise_transformer),
        ("predictor", LinearRegression()),
    ]
)
pl3 = Pipeline(
    [
        (
            "add_past_period_columns",
            FunctionTransformer(create_past_ask_Open_num_large),
        ),
        ("fill_na", fillna_transformer),
        ("normalise", normalise_transformer),
        ("predictor", LinearRegression()),
    ]
)

stack = StackingRegressor(
    [("small_lookback", pl1), ("medium_lookback", pl2), ("large_lookback", pl3)],
    final_estimator=LinearRegression(),
)

X = model_data[["ASK_OPEN"]]
y = model_data["r"]
stack.fit(X, y)
autolog_run = mlflow.last_active_run()

with mlflow.start_run(run_id=autolog_run.info.run_id) as run:
    for i in range(3):
        mlflow.log_metric(
            key="stack__final_estimator___coef_" + str(i),
            value=stack.final_estimator_.coef_[i],
        )
    fig, ax = plt.subplots()
    ax.scatter(stack.predict(X), y)
    mlflow.log_figure(fig, "predictions_actual_scatter.png")
    mlflow.log_dict(model_config, "model_config.json")
    mlflow.log_dict(historical_prices_config, "historical_prices_config.json")

print(f"Logged data and model in run {autolog_run.info.run_id}")

# if SAVE_MODEL:
#     logging.info("Saving model")
#     joblib.dump(stack, "model.pkl")
# else:
#     stack = joblib.load(Path(__file__).parent/'model.pkl')
