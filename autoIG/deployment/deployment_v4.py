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
from autoIG.create_data import write_to_transations_joined, whipe_data
from autoIG.deployment.deployment_config import models_to_deploy

# import sqlite3
import logging
from autoIG.utils import append_with_header
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
    # Get information for deployment from the model itself
    # client = MlflowClient()
    # mv = client.get_model_version(model_name, model_version)
    # model_run_id = mv.run_id
    # run = client.get_run(model_run_id)
    # run_data = run.data  # contains all run data
    # run_params = run_data.params
    # epic = run_params["epic"]
    past_periods_needed = int(
        run.data.params["past_periods_needed"]
    )  # These are used in on_update function
    target_periods_in_future = int(run.data.params["target_periods_in_future"])

    model = load_model(f"models:/{model_name}/{model_version}")
    write_stream_length(0)

    # Set up Subscription
    # Is this really needed? We've already set up a service outside this

    # ig_service = IGService(**ig_service_config)
    # ig_stream_service = IGStreamService(ig_service)
    # ig_stream_service.create_session()

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
        # We are resampling everytime time, this is inefficient
        # Dont take the last one since it is not complete yet.
        stream = (
            raw_stream.resample(pd.Timedelta(seconds=60), label="right")
            .last()
            .dropna()  # since if there is a gap in raw stream all the intermediate resamples are filled with nas
            .iloc[:-1, :]
        )
        stream.to_csv(TMP_DIR / "stream.csv", mode="w", header=True)
        # with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
        #     stream.to_sql(name="stream", con=sqliteConnection, if_exists="append")
        stream_length = stream.shape[0]

        # Only predict when there is a new piece of stream data
        if (stream_length >= past_periods_needed) and (
            read_stream_length() < stream_length
        ):
            # When a new row is added to stream we jump into action.
            # Do the folling:
            # 1. Update the stream length number, so we know when next to jump into action
            # 2. Make a new prediction
            # 3. Check if new prediction warrents buying and buy
            # 4. record information about the buy in position_metrics
            # 4. Check if now we need to sell anything

            # 1
            write_stream_length(stream_length)

            # 2
            predictions = model.predict(stream[["ASK_OPEN"]])  # predict on all stream
            latest_prediction = predictions[-1]
            logging.info(f"Latest prediction: {latest_prediction}")

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
                        "y_pred": [latest_prediction],
                        "buy_date": [pd.to_datetime(open_position_responce["date"])],
                        "sell_date": [
                            (
                                pd.to_datetime(open_position_responce["date"]).round(
                                    "1min"
                                )
                                + timedelta(minutes=target_periods_in_future)
                            )
                        ],
                        "buy_level_responce": [open_position_responce["level"]],
                        # We get this from transactions, but doulbe check
                        "sold": False,
                    }
                )
                to_sell = pd.DataFrame(  # get rid of this
                    {
                        # "dealreference": [resp["dealReference"]],
                        "dealId": [open_position_responce["dealId"]],
                        "sell_date": [  # change to sell_date
                            (
                                pd.to_datetime(open_position_responce["date"]).round(
                                    "1min"
                                )
                                + timedelta(minutes=target_periods_in_future)
                            )
                        ],
                        "sold": False,
                    }
                )
                # append_with_header(to_sell, "to_sell.csv")
                append_with_header(position_metrics, "position_metrics.csv")
                # with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
                #     position_metrics.to_sql(
                #         name="position_metrics",
                #         con=sqliteConnection,
                #         if_exists="append",
                #     )

                # append responce
                # This info is in activity??
                single_responce = (
                    pd.Series(open_position_responce).to_frame().transpose()
                )
            # 5
            # to_sell = pd.read_csv(TMP_DIR / "to_sell.csv")

            position_metrics = pd.read_csv(TMP_DIR / "position_metrics.csv")

            need_to_sell_bool = (
                pd.to_datetime(position_metrics.sell_date) < datetime.now()
            )
            sell_bool = need_to_sell_bool & (position_metrics.sold == False)

            def close_open_positions(s: pd.Series):
                """
                Takes a series of DealIds positions to close, and closes them.
                Updating the sold and 
                """
                for i in s:
                    logging.info(f"Closing a position with DealId: {i}")
                    # Q: How does close position responce differ from open position responce
                    close_position_responce = ig_service.close_open_position(
                        **close_position_config_(dealId=i)
                    )
                    sold = pd.DataFrame(
                        {
                            "dealId": [i],
                            "close_level_resp": close_position_responce["level"],
                        }
                    )
                    append_with_header(sold, "sold.csv")
                return None

            close_open_positions(position_metrics[sell_bool].dealId)

            # for i in position_metrics[sell_bool].dealId:
            #     logging.info(f"Closing a position {i}")
            #     close_position_responce = ig_service.close_open_position(
            #         **close_position_config_(dealId=i)
            #     )
            #     sold = pd.DataFrame(
            #         # We need this to get the closing refernce for the IG.transactions table
            #         {
            #             "dealId": [i],
            #             # "dealreference": [resp["dealReference"]],  # closing reference
            #             # "dealId": [close_position_responce["dealId"]],  # closing dealId, the same as i
            #             "close_level_resp": close_position_responce[
            #                 "level"
            #             ],  # These should come from IG.transactions, but just checking
            #             # "profit_resp": close_position_responce[
            #             #     "profit"
            #             # ],  # These should come from IG.transactions, but just checking
            #         }
            #     )
            #     append_with_header(sold, "sold.csv")
            #     # with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
            #     #     sold.to_sql(name="sold", con=sqliteConnection, if_exists="append")


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
                (current_time - deployment_start).total_seconds()
            )

            write_to_transations_joined(secs_ago=secs_since_deployment)

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
        # run_data = run.data  # contains all run data
        # run_params = run_data.params
        # epic = run_params["epic"]

        # model = load_model(f"models:/{model_name}/{model_version}")
        # write_stream_length(0)

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
    ig_stream_service.disconnect()
    return None


def deploy():
    run()
    # sell all open positions to clean up?
    return None


if __name__ == "__main__":

    deploy()
