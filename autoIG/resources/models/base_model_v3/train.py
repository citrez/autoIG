from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.models
import pandas as pd
from sklearn import set_config
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error

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

# MLflow config
EXPERIMENT_NAME = "stacked-linear-reg"
MODEL_NAME = "stacked-linear-reg-model"
mlflow.set_experiment(EXPERIMENT_NAME)
# mlflow.sklearn.autolog(
#     registered_model_name=MODEL_NAME,
#     log_input_examples=True,
# )
set_config(transform_output="pandas")

model_config = dict()
historical_prices_config_ig = dict()

model_config["RELOAD_DATA"] = False
model_config["SOURCE"] = Source.yahoo_finance.value
model_config[
    "EPIC"
] = Epics.US_CRUDE_OIL.value  # Could get both of these based on US_CRUDE_OIL and source
model_config["TICKER"] = Tickers.US_CRUDE_OIL.value

ig_yf_resolution = {"1min": "1m"}
resolutions = {"1min": {"IG": "1min", "YF": "1m"}}

historical_prices_config_ig["resolution"] = "1min"  # Currently not used?
historical_prices_config_ig["numpoints"] = 500

if model_config["RELOAD_DATA"]:
    if model_config["SOURCE"] == "IG":
        from trading_ig.rest import IGService

        from autoIG.config import ig_service_config

        ig_service = IGService(**ig_service_config)
        ig = ig_service.create_session()
        results_ = ig_service.fetch_historical_prices_by_epic(
            model_config["EPIC"], **historical_prices_config_ig
        )
        model_data = results_["prices"]
        model_data.to_pickle("model_data_ig.pkl")
    if model_config["SOURCE"] == "YF":
        import yfinance as yf

        ticker = yf.Ticker(model_config["TICKER"])
        start = "2022-12-24"
        end = "2022-12-31"  # only 7 days worth of 1m granulairty allow. TODO: Check for this
        model_data = ticker.history(interval="1m", start=start, end=end)
        model_data.to_pickle("model_data_yf.pkl")
    else:
        Exception("Please provide source to reload data from: (IG/YF)")

else:
    if model_config["SOURCE"] == "IG":
        model_data = pd.read_pickle(Path(__file__).parent / "model_data_ig.pkl")
    if model_config["SOURCE"] == "YF":
        model_data = pd.read_pickle(Path(__file__).parent / "model_data_yf.pkl")

from autoIG.modelling import create_future_bid_Open,generate_target
if model_config["SOURCE"] == "IG":
    model_data = model_data.pipe(adapt_IG_data_for_training).pipe(create_future_bid_Open).pipe(generate_target).dropna() # we need this to create the target)
if model_config["SOURCE"] == "YF":
    model_data = model_data.pipe(adapt_YF_data_for_training).pipe(create_future_bid_Open).pipe(generate_target).dropna()
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
# This is a shorter way
# lookbacks = [3,10,15]
# funcs =  (partial(create_past_ask_Open, num=n) for n in lookbacks)
# def create_pipeline(f):
#     return Pipeline(
#     [
#         (
#             "add_past_period_columns",
#             FunctionTransformer(f),
#         ),
#         ("fill_na", fillna_transformer),
#         ("normalise", normalise_transformer),
#         ("predictor", LinearRegression()),
#     ]
# )
# pipelines = (create_pipeline(f) for f in funcs)
# pipeline_names = (str(n)+"_lookback" for n in lookbacks)
# tuple(zip(pipeline_names,pipelines))

stack = StackingRegressor(
    [("small_lookback", pl1), ("medium_lookback", pl2), ("large_lookback", pl3)],
    final_estimator=LinearRegression(),
)

TEST_SIZE = 100
train = model_data.iloc[TEST_SIZE:, :]
test = model_data.iloc[0:TEST_SIZE, :] # From the begining, we want the best data to be used in training
X_train = train[["ASK_OPEN"]]
y_train = train["r"]
X_test = test[["ASK_OPEN"]]
y_test = test["r"]
stack.fit(X_train, y_train)
# autolog_run = mlflow.last_active_run()
with mlflow.start_run( 
    # Uses experiment set in mlflow.set_experiment
    # run_id= autolog_run.info.run_id
    description = 'This is a description of the model run'
    ) as run:
    model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"
    # mlflow.register_model(model_uri=model_uri,name='stacked-linear-reg-model')
    mlflow.sklearn.log_model(
        sk_model=stack,
        artifact_path="sklearn-model",
        registered_model_name=MODEL_NAME,
        input_example=X_train.iloc[0:3,:],
        signature = mlflow.models.infer_signature(X_train.iloc[0:5,:],stack.predict(X_train.iloc[0:5,:]))
    )

    for i in range(3):
        mlflow.log_metric("stack__final_estimator___coef_" + str(i),stack.final_estimator_.coef_[i],)
    fig, ax = plt.subplots()
    ax.scatter(y_train,stack.predict(X_train))
    plt.xlabel('actual')
    plt.ylabel('prediction')
    mlflow.log_figure(fig, "training_predictions_scatter.png")

    fig, ax = plt.subplots()
    ax.scatter( y_test,stack.predict(X_test))
    plt.xlabel('actual')
    plt.ylabel('prediction')
    mlflow.log_figure(fig, "testing_predictions_scatter.png")

    fig, ax = plt.subplots()
    bins = int(len(y_train)*0.01)
    plt.hist([y_train,stack.predict(X_train)],bins = bins, alpha = 0.5,label = ['actual','predictions'])
    plt.legend(loc='upper right')
    mlflow.log_figure(fig, "predicted_histogram.png")

    fig, ax = plt.subplots()
    ax.scatter(y_train,(y_train - stack.predict(X_train) ) )
    mlflow.log_figure(fig, "training_error_size.png")

    fig, ax = plt.subplots()
    ax.scatter(y_test,y_test - stack.predict(X_test))
    mlflow.log_figure(fig, "testing_error_size.png")

    mlflow.log_metric('training_frequency', (stack.predict(X_train)>1.01).sum()/ len(X_train) )
    mlflow.log_metric('testing_frequency', (stack.predict(X_test)>1.01).sum()/ len(X_test) )
    mlflow.log_param('final_estimator',stack.final_estimator)
    mlflow.log_dict(model_config, "model_config.json")
    mlflow.log_dict(historical_prices_config_ig, "historical_prices_config.json")
    stack.score(X_train,y_train) # Just calling this, mlflow autologger logs it  
    stack.score(X_test,y_test) # Just calling this, mlflow autologger logs it 
    mlflow.log_metric('testing_mean_absoloute_error',mean_absolute_error(y_pred=stack.predict(X_test),y_true = y_test) )# Just calling this, mlflow autologger logs it 



print(f"Logged data and model in run {run.info.run_id}")
