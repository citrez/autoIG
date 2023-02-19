"""
This script deploys the models, so that they start buying and selling.
Ideallly, this should be model agnostic. 
The only thing that should matter is the data feed the model needs.
i.e prices only models should all work with the same deployment script. 
"""
from datetime import datetime
from mlflow import MlflowClient
from autoIG.utils import (
    prices_stream_responce,
    read_from_tmp,
    TMP_DIR,
    read_stream_length,
    write_stream_length,
)
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

# from mlflow.pyfunc import load_model

# TODO: Sort out logger handling
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",  # filemode='w', filename='log.log'
)

######
# WE DO NOT KEEP ANY OLD DATA EACH TIME WE DEPLOY.
# THIS IS BY CHOICE UNTIL WE HAVE A MORE MATURE PRODUCT
whipe_data()
#######


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
    # write_stream_length(0)

    deployment_start = datetime.now()

    def on_update(item):
        """
        Everytime the subscription get new data, this is run

        1. Load and save in IG price stream
        2. Resample the raw stream of subscription.
        Need to get it in the form needed for predictions. 1 row per min
        3. Everytime the stream gets larger, make a prediction.
        4. If the prediction is greater than the threshold, BUY log information (including when to sell in position_metrics)
        5. Check which dealIds need selling

        """

        if item["values"]["MARKET_STATE"] != "TRADEABLE":
            raise Exception(
                f"Market not open for trading. Market state: {item['values']['MARKET_STATE']}"
            )

        # TODO: @citrez #4  Change raw_stream name to price_stream
        # TODO: #3 Change stream to price_stream_resampled

        append_with_header(prices_stream_responce(item), "raw_stream.csv")
        raw_stream = read_from_tmp("raw_stream.csv")
        stream_max_updated_at = read_from_tmp(
            "stream.csv", usecols=["UPDATED_AT"]
        ).index.max()
        raw_stream_max_updated_at = raw_stream.index.max()

        # Only bother updating stream if we have enough data from raw_stream
        if (stream_max_updated_at < raw_stream_max_updated_at.replace(second=0)) or (
            stream_max_updated_at is pd.NaT
        ):
            # We are resampling everytime time, this is inefficient
            # Dont take the last one since it is not complete yet.
            stream = (
                raw_stream.resample(pd.Timedelta(seconds=60), label="right")
                .last()
                .dropna()  # since if there is a gap in raw stream all the intermediate resamples are filled with nas
                .iloc[:-1, :]
            )
            stream.to_csv(TMP_DIR / "stream.csv", mode="w", header=True)
            stream_length = stream.shape[0]

            # write_stream_length(stream_length)
            # try:
            #     stream = read_from_tmp("stream.csv")
            # except pd.errors.EmptyDataError:
            #     stream_length = 0
            # with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
            #     stream.to_sql(name="stream", con=sqliteConnection, if_exists="append")

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
                latest_prediction = predictions[-1]
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
                    # resp in columns in confirms.create_open_positiion

                    logging.info(
                        f"Opened position with DealId: {open_position_responce['dealId']}. Status: { open_position_responce['dealStatus'] }"
                    )
                    # 4
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
                            "sold": False,
                            "y_pred": [latest_prediction],
                        }
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

                # There is no need for this mess. We can have a sell_data and a sold table
                # And do an anti join on sold and a sell_data<now to check which havent been sold
                # and need to be. Dont like this updating on state in the position metrics table
                # !

                need_to_sell_bool = (
                    pd.to_datetime(position_metrics.sell_date) < datetime.now()
                )
                sell_bool = need_to_sell_bool & (position_metrics.sold == False)

                close_open_positions(
                    position_metrics[sell_bool].dealId, ig_service=ig_service
                )

                # This assumes that those that we needed to sell were succesfully sold
                position_metrics.sold = need_to_sell_bool
                position_metrics.to_csv(
                    TMP_DIR / "position_metrics.csv", mode="w", header=True, index=False
                )

                # with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
                #     # Maybe open and close connection only once at the begining and end?
                #     # Or wrap everthing in a context manager
                #     position_metrics.to_sql(name="position_metrics", con=sqliteConnection, if_exists="append")

                secs_since_deployment = int(
                    (datetime.now() - deployment_start).total_seconds()
                )

                # write_to_transations_joined(secs_ago=secs_since_deployment)
                position_metrics = pd.read_csv(TMP_DIR / "position_metrics.csv")
                try:
                    sold = pd.read_csv(TMP_DIR / "sold.csv")
                except pd.errors.EmptyDataError:
                    print("Nothing sold yet, creating empty dataframe")
                    sold = pd.DataFrame(columns=["dealId", "close_level_responce"])

                position_metrics_merged = position_metrics.merge(sold, how="left")

                position_metrics_merged["y_true"] = (
                    position_metrics_merged["close_level_responce"]
                    / position_metrics_merged["buy_level_responce"]
                )
                position_metrics_merged["y_pred_actual"] = (
                    position_metrics_merged["y_pred"]
                    * position_metrics_merged["buy_level_responce"]
                )
                position_metrics_merged["profit_responce"] = np.where(
                    position_metrics_merged["close_level_responce"].isna(),
                    np.NaN,
                    1
                    * (
                        position_metrics_merged["close_level_responce"]
                        - position_metrics_merged["buy_level_responce"]
                    ),
                )
                position_metrics_merged = position_metrics_merged.fillna("None")
                position_metrics_merged.to_csv(
                    TMP_DIR / "position_metrics_merged.csv", index=False
                )

            return None

    return on_update


def run():

    for i in models_to_deploy:
        model_name, model_version, r_threshold = (
            i["model_name"],
            i["model_version"],
            i["r_threshold"],
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
        print(f"{model_name}, {model_version}, {r_threshold}")

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


def deploy():
    run()

    return None


if __name__ == "__main__":

    deploy()
