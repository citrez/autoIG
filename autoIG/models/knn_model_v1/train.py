from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.models
import pandas as pd
from sklearn import set_config
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error

from autoIG.modelling import create_future_bid_Open, generate_target
from autoIG.modelling import (
    adapt_IG_data_for_training,
    adapt_YF_data_for_training,
    create_past_ask_Open,
    fillna_,
    normalise_,
)
from autoIG.utils import print_shape

set_config(transform_output="pandas")

# MLflow config
EXPERIMENT_NAME = "knn-reg"
MODEL_NAME = "knn-reg-model"
mlflow.set_experiment(EXPERIMENT_NAME)

reload_data_config = dict()
# ig_yf_resolution = {"1min": "1m"}
# resolutions = {"1min": {"IG": "1min", "YF": "1m"}}
reload_data_config["resolution"] = "1min"  # Currently not used?
reload_data_config["numpoints"] = 500

from model_config import source, epic, ticker

MLFLOW_RUN = True


def get_data(reload_data=False):
    """
    This gets, or reloads cached data for prices.
    Prices either come from IG, or yahoo finance.
    """

    if not reload_data:
        if source == "IG":
            model_data = pd.read_pickle(
                Path(__file__).parent / "model_data/model_data_ig.pkl"
            )
        if source == "YF":
            model_data = pd.read_pickle(
                Path(__file__).parent / "model_data/model_data_yf.pkl"
            )

    if reload_data:
        if source == "IG":
            from trading_ig.rest import IGService
            from autoIG.config import ig_service_config

            ig_service = IGService(**ig_service_config)
            _ = ig_service.create_session()
            results_ = ig_service.fetch_historical_prices_by_epic(epic, **reload_data)
            model_data = results_["prices"]
            model_data.to_pickle("model_data/model_data_ig.pkl")

        if source == "YF":
            import yfinance as yf

            tick = yf.Ticker(ticker=ticker)
            start = "2022-12-24"
            end = "2022-12-31"  # only 7 days worth of 1m granulairty allow. TODO: Check for this
            model_data = tick.history(interval="1m", start=start, end=end)
            model_data.to_pickle("model_data/model_data_yf.pkl")
        else:
            Exception("Please provide source to reload data from: (IG/YF)")

    return model_data


model_data = get_data()


def adapt_data(d_: pd.DataFrame):
    d = d_.copy()

    if source == "IG":
        d = (
            d.pipe(adapt_IG_data_for_training)
            .pipe(create_future_bid_Open)
            .pipe(generate_target)
            .dropna()
        )  # we need this to create the target
    if source == "YF":
        d = (
            d.pipe(adapt_YF_data_for_training)
            .pipe(create_future_bid_Open)
            .pipe(generate_target)
            .dropna()
        )
    d.pipe(print_shape)
    return d


model_data = adapt_data(model_data)

create_past_ask_Open_num_small = partial(create_past_ask_Open, num=5)

fillna_transformer = FunctionTransformer(fillna_)
normalise_transformer = FunctionTransformer(normalise_)

pl = Pipeline(
    [
        (
            "add_past_period_columns",
            FunctionTransformer(create_past_ask_Open_num_small),
        ),
        ("fill_na", fillna_transformer),
        ("normalise", normalise_transformer),
        ("predictor", KNeighborsRegressor()),
    ]
)

TEST_SIZE = 100
train = model_data.iloc[TEST_SIZE:, :]
test = model_data.iloc[
    0:TEST_SIZE, :
]  # From the begining, we want the best data to be used in training
X_train = train[["ASK_OPEN"]]
y_train = train["r"]
X_test = test[["ASK_OPEN"]]
y_test = test["r"]
pl.fit(X_train, y_train)
# autolog_run = mlflow.last_active_run()
if MLFLOW_RUN:
    with mlflow.start_run(
        # Uses experiment set in mlflow.set_experiment
        # run_id= autolog_run.info.run_id
        description="This is a description of the model run"
    ) as run:
        # Log model
        model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"
        mlflow.sklearn.log_model(
            sk_model=pl,
            artifact_path="sklearn-model",  # in the run, model artifacts are stored in artifacts/artifact_path
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[0:3, :],
            signature=mlflow.models.infer_signature(
                X_train.iloc[0:5, :], pl.predict(X_train.iloc[0:5, :])
            ),
        )

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.scatter(y_train, pl.predict(X_train))
        ax1.set_title("train")
        ax2.scatter(y_test, pl.predict(X_test))
        ax2.set_title("test")
        plt.xlabel("actual")
        plt.ylabel("prediction")
        plt.suptitle("training_and_testing_predictions_scatter")

        mlflow.log_figure(fig, "training_and_testing_predictions_scatter.png")

        fig, ax = plt.subplots()
        bins = int(len(y_train) * 0.01)
        plt.hist(
            [y_train, pl.predict(X_train)],
            bins=bins,
            alpha=0.5,
            label=["actual", "predictions"],
        )
        plt.legend(loc="upper right")
        plt.suptitle("predicted_histogram")

        mlflow.log_figure(fig, "predicted_histogram.png")

        fig, ax = plt.subplots(2)
        ax[0].scatter(y_train, (y_train - pl.predict(X_train)))
        ax[0].set_title("train")
        ax[1].scatter(y_test, y_test - pl.predict(X_test))
        ax[1].set_title("test")
        plt.suptitle("training_error_size")
        mlflow.log_figure(fig, "training_and_testing_error_size.png")

        mlflow.log_metric(
            "training_frequency", (pl.predict(X_train) > 1.01).sum() / len(X_train)
        )
        mlflow.log_metric(
            "testing_frequency", (pl.predict(X_test) > 1.01).sum() / len(X_test)
        )
        mlflow.log_param("source", source)
        mlflow.log_param("epic", epic)
        mlflow.log_param("ticker", ticker)
        mlflow.log_dict(reload_data_config, "historical_prices_config.json")

        mlflow.log_metric(
            "testing_mae",
            mean_absolute_error(y_pred=pl.predict(X_test), y_true=y_test),
        )  
        mlflow.log_metric(
            "training_mae",
            mean_absolute_error(y_pred=pl.predict(X_test), y_true=y_train),
        )

        ## KNN specific metrics
        

    print(f"Logged data and model in run {run.info.run_id}")
