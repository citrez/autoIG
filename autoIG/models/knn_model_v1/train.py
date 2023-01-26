from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.models
import pandas as pd
import numpy as np
from sklearn.utils import estimator_html_repr
from sklearn import set_config

set_config(transform_output="pandas")
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from autoIG.modelling import create_future_bid_Open, generate_target
from autoIG.utils import DATA_DIR
from autoIG.modelling import (
    adapt_IG_data_for_training,
    adapt_YF_data_for_training,
    create_past_ask_Open,
    fillna_,
    normalise_,
)
from autoIG.utils import print_shape


MLFLOW_RUN = True
# MLflow config
EXPERIMENT_NAME = "knn-reg"
MODEL_NAME = "knn-reg-model"
mlflow.set_experiment(EXPERIMENT_NAME)

from model_config import (
    source,
    epic,
    ticker,
    threshold,
    past_periods_needed,
    target_periods_in_future,
    resolution,
)

# The fetching of the data from IG is done in another script.
# Do a load of local data.
# We could do a check on local data to check we have what we need, and if
# We dont get an error to fetch the data


# TODO: Now we are doing this from local, we should probably set up some DVC data tracking
model_data = pd.read_csv(
    DATA_DIR
    / "training"
    / source
    / ticker.replace(".", "_")
    / resolution
    / "full_data.csv",
    parse_dates=["datetime"],  # ,dtype={'datetime':np.datetime64}
)

# as a rule, dont use massive function
# Do little chucks and pipe
model_data = (
    model_data.pipe(adapt_YF_data_for_training)
    .pipe(create_future_bid_Open)
    .pipe(generate_target)
    .dropna()
)
model_data.pipe(print_shape)


def create_pipeline():

    past_periods = 5
    assert past_periods <= past_periods_needed
    create_past_ask_Open_num_small = partial(create_past_ask_Open, past_periods=5)
    # fillna_transformer = FunctionTransformer(fillna_)
    fillna_transformer = SimpleImputer(strategy="constant", fill_value=-999)
    fillna_transformer.set_output(transform="pandas")

    normalise_transformer = FunctionTransformer(normalise_)
    knn_params = {"n_neighbors": 15}
    pl = Pipeline(
        [
            (
                "add_past_period_columns",
                FunctionTransformer(create_past_ask_Open_num_small),
            ),
            ("fill_na", fillna_transformer),
            ("normalise", normalise_transformer),
            ("predictor", KNeighborsRegressor(**knn_params)),
        ]
    )
    if MLFLOW_RUN:
        with mlflow.start_run():
            mlflow.log_params(params=knn_params)
    return pl


pl = create_pipeline()

X = model_data[["ASK_OPEN"]]
y = model_data["r"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, shuffle=False)
# How do we do this from the begining, we want the best (most recent) data to be used in training

print(pd.Series(X_train.index).describe(datetime_is_numeric=True))
print(pd.Series(X_test.index).describe(datetime_is_numeric=True))

pl.fit(X_train, y_train)

X_train_transformed = pl[:-1].transform(X_train)

# Look at the k nearest
random_training_sample = X_train_transformed.iloc[54:55, :]
distance_to, indecies_of = pl[-1].kneighbors(random_training_sample)
indecies_of = indecies_of[0]
distance_to = distance_to[0]
nearest_of_random_training_sample = X_train_transformed.iloc[indecies_of, :]
nearest_of_random_training_sample = nearest_of_random_training_sample.assign(
    distance_to=distance_to
)


if MLFLOW_RUN:
    with mlflow.start_run(
        # Uses experiment set in mlflow.set_experiment
        run_id=mlflow.last_active_run().info.run_id,
        description="This is a description of the model run",
    ) as run:
        # Log model
        # Log the actuall model object in the sklearn-model/ directory in artufacts
        # Also, seperately, register this model
        mlflow.sklearn.log_model(
            sk_model=pl,
            # In the run, model artifacts are stored in artifacts/artifact_path
            artifact_path="sklearn-model",
            # registered_model_name=MODEL_NAME,
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
        plt.xlabel("y_true")
        plt.ylabel("y_preds")
        # TODO: add the optimal line
        plt.suptitle("training_and_testing_predictions_scatter")
        mlflow.log_figure(fig, "training_and_testing_predictions_scatter.png")
        plt.close()

        fig, ax = plt.subplots()
        bins = int(len(y_train) * 0.003)
        plt.hist(
            [y_train, pl.predict(X_train)],
            bins=bins,
            alpha=0.5,
            range=(0.999, 1.001),
            label=["y_true", "y_pred"],
        )
        plt.legend(loc="upper right")
        plt.suptitle("training_y_true y_preds")
        mlflow.log_figure(fig, "training_y_true_y_preds.png")
        plt.close()

        fig, (ax0,ax1) = plt.subplots(2)
        ax0.scatter(y_train, (y_train - pl.predict(X_train)).abs() )
        ax0.set_title("train")
        ax1.scatter(y_test, (y_test - pl.predict(X_test)).abs() )
        ax1.set_title("test")
        ax0.set_xlabel('t_true')
        ax0.set_ylabel('absoloute error')
        plt.suptitle("training_error_size")
        mlflow.log_figure(fig, "absoloute_error_size.png")
        plt.close()

        mlflow.log_metric(
            "training_frequency", (pl.predict(X_train) > threshold).sum() / len(X_train)
        )
        mlflow.log_metric(
            "testing_frequency", (pl.predict(X_test) > threshold).sum() / len(X_test)
        )
        # Log parameters of model
        mlflow.log_param("source", source)
        mlflow.log_param("epic", epic)
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("past_periods_needed", past_periods_needed)
        mlflow.log_param("target_periods_in_future", target_periods_in_future)

        # mlflow.log_dict(
        #     reload_data_config,
        #     "historical_prices_config.json",
        # )

        mlflow.log_metric(
            "testing_mae",
            mean_absolute_error(y_pred=pl.predict(X_test), y_true=y_test),
        )
        mlflow.log_metric(
            "training_mae",
            mean_absolute_error(y_pred=pl.predict(X_train), y_true=y_train),
        )
        CURRENT_DIR = Path(__file__).parent
        with open(CURRENT_DIR / "pl.html", "w") as f:
            f.write(estimator_html_repr(pl))
        mlflow.log_artifact(local_path=CURRENT_DIR / "pl.html", artifact_path="docs")

        ## KNN specific metrics

        mlflow.log_metric(
            "testing_neigh_dist_n_neighbors_1",
            pl.named_steps.predictor.kneighbors(
                pl[:-1].transform(X_test), n_neighbors=1
            )[0].sum(),
        )
        mlflow.log_metric(
            "training_neigh_dist_n_neighbors_1",
            pl.named_steps.predictor.kneighbors(
                pl[:-1].transform(X_train), n_neighbors=1
            )[0].sum(),
        )

        mlflow.log_metric(
            "testing_neigh_dist_n_neighbors_5",
            pl.named_steps.predictor.kneighbors(
                pl[:-1].transform(X_test), n_neighbors=5
            )[0].sum(),
        )
        mlflow.log_metric(
            "training_neigh_dist_n_neighbors_5",
            pl.named_steps.predictor.kneighbors(
                pl[:-1].transform(X_train), n_neighbors=5
            )[0].sum(),
        )

        mlflow.log_metric(
            "testing_neigh_dist_n_neighbors_10",
            pl.named_steps.predictor.kneighbors(
                pl[:-1].transform(X_test), n_neighbors=10
            )[0].sum(),
        )
        mlflow.log_metric(
            "training_neigh_dist_n_neighbors_10",
            pl.named_steps.predictor.kneighbors(
                pl[:-1].transform(X_train), n_neighbors=10
            )[0].sum(),
        )

        fig, ax = plt.subplots()
        bins = int(len(y_train) * 0.01)
        plt.hist(
            [
                pl.named_steps.predictor.kneighbors(
                    pl[:-1].transform(X_train), n_neighbors=5
                )[0].sum(axis=1),
                pl.named_steps.predictor.kneighbors(
                    pl[:-1].transform(X_test), n_neighbors=5
                )[0].sum(axis=1),
            ],
            bins=bins,
            alpha=0.5,
            range=(0, 0.003),
            label=["X_train", "X_test"],
            density = True
        )
        plt.legend(loc="upper right")
        # plt.suptitle("training_y_true y_preds")
        mlflow.log_figure(fig, "closeness_hist.png")
        plt.close()

        # If the test set is close to train set, does it produce better predictions
        X_test_transformed = pl[:-1].transform(X_test)
        # The distances to the X_train
        # Q: Is the shorter the distances the better the prediction is?
        X_test_distance_to = pl[-1].kneighbors(X_test_transformed)[0].sum(axis=1)
        absoloute_error = np.array((pl.predict(X_test) - y_test).abs())

        fig, ax = plt.subplots()
        ax.scatter(X_test_distance_to, absoloute_error, s=1, alpha=0.8)
        ax.set_xlim([0, 0.02])
        ax.set_xlabel("distance_to")
        ax.set_ylabel("absoloute_error")
        ax.set_title("X_test")
        mlflow.log_figure(fig,"closeness_vrs_error.png")
        plt.close('all')

    print(f"Logged data and model in run {run.info.run_id}")
