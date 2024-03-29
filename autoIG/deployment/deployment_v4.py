"""
This script deploys the models, so that they start buying and selling.
Ideallly, this should be model agnostic. 
The only thing that should matter is the data feed the model needs.
i.e prices only models should all work with the same deployment script. 
"""
from datetime import datetime
from mlflow import MlflowClient
from autoIG.utils import prices_stream_responce, read_from_tmp, TMP_DIR
import numpy as np
from autoIG.create_data import whipe_data
from autoIG.deployment.deployment_config import models_to_deploy

# import sqlite3
import logging
from autoIG.utils import append_with_header, close_open_positions
from trading_ig import IGService, IGStreamService
from trading_ig.lightstreamer import Subscription
from autoIG.config import (
    open_position_config_,
    close_position_config_,
    ig_service_config,
)

import pandas as pd
from datetime import timedelta
from mlflow.sklearn import load_model  # this returns the actual sklearn model

# from mlflow.pyfunc import load_model # This returns a consistent mlflow model thing


################################################
# WE DO NOT KEEP ANY OLD DATA EACH TIME WE DEPLOY.
# THIS IS BY CHOICE UNTIL WE HAVE A MORE MATURE PRODUCT
whipe_data()
################################################


def wrap(model_name, model_version, r_threshold, run, ig_service):
    """
    On update only takes one arguemnt. So we use a closure to define the local
    variables that are being fed into on_update
    """
    past_periods_needed = int(
        run.data.params["past_periods_needed"]
    )  # These are used in on_update function
    target_periods_in_future = int(run.data.params["target_periods_in_future"])

    pipeline = load_model(f"models:/{model_name}/{model_version}")

    deployment_start = datetime.now()

    def on_update(item):
        """
        Everytime the subscription gets new price data, this is run.

        1. Check if the market is open and tradable
        2. Load and save the IG raw stream
        3. Resample the raw stream of subscription to a stream table.
        Need to get it in the form seen in training, 1 row per min
        4. Everytime the stream gets larger, make a prediction using the past_periods needed.
        5. If the prediction is greater than the threshold, BUY log information (including when to sell in position_metrics)
        6. Check which dealIds need selling

        """

        if item["values"]["MARKET_STATE"] != "TRADEABLE":
            raise Exception(
                f"Market not open for trading. Market state: {item['values']['MARKET_STATE']}"
            )

        # TODO: @citrez #4  Change raw_stream name to price_stream
        # TODO: #3 Change stream to price_stream_resampled

        append_with_header(prices_stream_responce(item), "raw_stream.csv")
        raw_stream = read_from_tmp("raw_stream.csv")
        raw_stream = raw_stream.set_index("UPDATED_AT")
        raw_stream.index = pd.to_datetime(raw_stream.index)
        raw_stream_max_updated_at = raw_stream.index.max()

        stream = read_from_tmp("stream.csv")
        # What if there is no stream.csv?

        if len(stream) == 0:
            stream_max_updated_at = pd.NaT
        else:
            stream = stream.set_index("UPDATED_AT")
            stream.index = pd.to_datetime(stream.index)
            stream_max_updated_at = stream.index.max()

        # Only bother updating stream if we have enough data from raw_stream so that resampling will create a new line in stream
        # Or if there is nothing in stream, since then we want to keep on resamlping, as we are dropping the last line, we will only get a stream when we have it
        if (stream_max_updated_at < raw_stream_max_updated_at.replace(second=0)) or (
            stream_max_updated_at is pd.NaT
        ):
            # We are resampling everytime time, this is inefficient
            # Dont take the last one since it is not complete yet.
            stream = (
                raw_stream.resample(pd.Timedelta(seconds=60), label="right")
                .last()
                .dropna()  # since if there is a gap in raw stream all the intermediate resamples are filled with nas
                .iloc[
                    :-1, :
                ]  # We dont take the last minutes resample, since it is incomplete.
            )
            stream.to_csv(TMP_DIR / "stream.csv", mode="w", header=True)
            stream_length = stream.shape[0]

            # Only predict when there is a new piece of stream data
            if stream_length >= past_periods_needed:
                logging.info("level 2")

                # When a new row is added to stream we jump into action.
                # Do the folling:
                # 1. Update the stream length number, so we know when next to jump into action
                # 2. Make a new prediction
                # 3. Check if new prediction warrents buying and buy
                # 4. record information about the buy in position_metrics
                # 4. Check if now we need to sell anything

                # 1
                # write_stream_length(stream_length)

                # 2
                predictions = pipeline.predict(
                    stream[["ASK_OPEN"]].iloc[-(past_periods_needed + 1) :, :]
                )  # predict on all stream
                latest_prediction = predictions[0]
                logging.info(f"Latest prediction: {latest_prediction}")
                if model_name == "knn-reg-model":
                    pass
                    # Get specific metrics from certain models that not all models expose
                    # Perhaps think of adding this to a dictionary in position_metrics as a dictionary

                    # distance_to, indecies_of = pipeline[-1].kneighbors(pipeline[:-1].transform(stream[["ASK_OPEN"]][-(past_periods_needed+1):,:] ))
                    # indecies_of = indecies_of[0]
                    # distance_to = distance_to[0]

                # 3
                if latest_prediction > r_threshold:
                    logging.info("BUY!")
                    open_position_responce = ig_service.create_open_position(
                        **open_position_config_(epic=run.data.params["epic"])
                    )
                    # responce in columns in confirms.create_open_positiion

                    logging.info(
                        f"Opened position with DealId: {open_position_responce['dealId']}. Status: { open_position_responce['dealStatus'] }"
                    )
                    # 4
                    # These are all the things we want to store about the prediction
                    position_metrics = pd.DataFrame(
                        {
                            # "dealreference": [resp["dealReference"]],
                            "dealId": [open_position_responce["dealId"]],
                            "model_used": [f"{model_name}-v{model_version}"],
                            "buy_date": [
                                pd.to_datetime(open_position_responce["date"])
                            ],
                            "sell_date": [
                                (
                                    pd.to_datetime(
                                        open_position_responce["date"]
                                    ).round("1min")
                                    + timedelta(minutes=target_periods_in_future)
                                )
                            ],
                            "buy_level_responce": [open_position_responce["level"]],
                            # We get this from transactions, but doulbe check
                            "y_pred": [latest_prediction],
                        }
                    )

                    position_metrics["y_pred_actual"] = (
                        position_metrics["y_pred"]
                        * position_metrics["buy_level_responce"]
                    )
                    # knn_metrics
                    append_with_header(position_metrics, "position_metrics.csv")

                    # append responce
                    # This info is in activity??
                    single_responce = (
                        pd.Series(open_position_responce).to_frame().transpose()
                    )
                # 5
                # to_sell = pd.read_csv(TMP_DIR / "to_sell.csv")

                position_metrics = pd.read_csv(TMP_DIR / "position_metrics.csv")

                # There is no need for this mess. We can have a sell_date and a sold table
                # And do an anti join on sold and a sell_data<now to check which havent been sold
                # and need to be. Dont like this updating on state in the position metrics table
                # !

                need_to_sell_bool = (
                    pd.to_datetime(position_metrics.sell_date) < datetime.now()
                )
                # TODO: Creak out the read_csv if empty return an empty dataframe whole thing into a simple function.

                sold = read_from_tmp(
                    "sold.csv", df_columns=["dealId", "close_level_responce"]
                )

                _ = position_metrics.loc[need_to_sell_bool, :].merge(
                    sold, on="dealId", how="left", indicator=True
                )
                # performs an anti join, looking for the ones we should have sold (sell_data< now)
                # that are not in the list of dealids we have sold in sold.csv
                need_to_sell = _[_._merge == "left_only"]
                close_open_positions(need_to_sell.dealId, ig_service=ig_service)

                # This assumes that those that we needed to sell were succesfully sold
                # position_metrics.sold = need_to_sell_bool
                # position_metrics.to_csv(
                #     TMP_DIR / "position_metrics.csv", mode="w", header=True, index=False
                # )

                # with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
                #     # Maybe open and close connection only once at the begining and end?
                #     # Or wrap everthing in a context manager
                #     position_metrics.to_sql(name="position_metrics", con=sqliteConnection, if_exists="append")

                position_metrics = pd.read_csv(TMP_DIR / "position_metrics.csv")

                sold = read_from_tmp(
                    "sold.csv", df_columns=["dealId", "close_level_responce"]
                )

                # TODO: do not duplicate positoin_metrics information
                # Only include information that we can only get from joining
                # position_metrics to closing price data
                position_outcomes = position_metrics.merge(
                    sold, how="left", on="dealId"
                )

                position_outcomes["y_true"] = (
                    position_outcomes["close_level_responce"]
                    / position_outcomes["buy_level_responce"]
                )
                # position_outcomes["y_pred_actual"] = (
                #     position_outcomes["y_pred"]
                #     * position_outcomes["buy_level_responce"]
                # )
                position_outcomes["profit_responce"] = np.where(
                    position_outcomes["close_level_responce"].isna(),
                    np.NaN,
                    1
                    * (
                        position_outcomes["close_level_responce"]
                        - position_outcomes["buy_level_responce"]
                    ),
                )
                position_outcomes['profit_responce_cumsum'] = position_outcomes['profit_responce'].cumsum()
                position_outcomes = position_outcomes.fillna("None")
                position_outcomes = position_outcomes[
                    ["dealId", "y_true", "profit_responce"]
                ]
                position_outcomes.to_csv(TMP_DIR / "position_outcomes.csv", index=False)
                # Do transformations that should utimately be views, defined in some transofmration layer like dbt
                # For now we can load in and save tables here and decide on naming etc
                grafana_deployment = raw_stream.join(
                    stream, how="left", lsuffix="_raw_stream", rsuffix="_stream"
                ).fillna("None")
                grafana_deployment.to_csv(TMP_DIR / "grafana_deployment.csv")
                grafana_transactions = position_metrics.merge(
                    position_outcomes, how="left", on="dealId"
                ).merge(sold, how=  'left',on = 'dealId').fillna("None")
                grafana_transactions.to_csv(TMP_DIR / "grafana_transactions.csv")

            return None

    return on_update


def run():

    for i in models_to_deploy:
        model_name, model_version, r_threshold = (
            i["model_name"],
            i["model_version"],
            i["r_threshold"],
        )
        logging.info(
            f"Deploying model: {model_name}\nVersion: {model_version}\nThreshold: {r_threshold}"
        )
        # Get information for deployment from the model itself
        client = MlflowClient()
        mv = client.get_model_version(model_name, model_version)
        model_run_id = mv.run_id
        run = client.get_run(model_run_id)

        # Set up Subscription
        ig_service = IGService(**ig_service_config)
        ig_stream_service = IGStreamService(ig_service)
        ig_stream_service.create_session()

        # deployment_start = datetime.now()
        sub = Subscription(
            mode="MERGE",
            items=["L1:" + run.data.params["epic"]],
            fields=["UPDATE_TIME", "BID", "OFFER", "MARKET_STATE"],
        )

        on_update = wrap(model_name, model_version, r_threshold, run, ig_service)
        sub.addlistener(on_update)
        ig_stream_service.ls_client.subscribe(sub)
    while True:
        user_input = input("Enter dd to termiate: ")
        if user_input == "dd":
            break

    # Total clean up
    open_positions = ig_service.fetch_open_positions()
    close_open_positions(open_positions.dealId, ig_service)
    ig_stream_service.disconnect()
    return None


def deployment():
    run()

    return None


if __name__ == "__main__":
    # TODO: Sort out logger handling
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",  # filemode='w', filename='log.log'
    )

    deployment()
