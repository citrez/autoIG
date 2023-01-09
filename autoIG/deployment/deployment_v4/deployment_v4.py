"""
This script deploys the model used. Ideallly, this should be model agnostic. 
The only thing that should matter is the data feed the model needs.
i.e prices only models should all work with the same deployment script. 
"""
from datetime import datetime
from autoIG.utils import (
    prices_stream_responce,
    read_stream,
    TMP_DIR,
    read_stream_length,
    write_stream_length,
)
from autoIG.create_data import write_to_transations_joined
import sqlite3
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

# from mlflow.sklearn import load_model
from mlflow.pyfunc import load_model

# Create a custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel('INFO')

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",  # filemode='w', filename='log.log'
)

from deployment_config import (
    model_name,
    model_version,
    r_threshold,
    epic,
    stream_length_needed,
    close_after_x_mins,
)

model = load_model(f"models:/{model_name}/{model_version}")
write_stream_length(0)

# Set up Subscription
ig_service = IGService(**ig_service_config)
ig_stream_service = IGStreamService(ig_service)
ig_stream_service.create_session()

# autoIG_config is the config needed for deployment
# the less the better, should be able to pick up a model and run
# autoIG_config = dict()
deployment_start = datetime.now()

sub = Subscription(
    mode="MERGE",
    items=["L1:" + epic],
    fields=["UPDATE_TIME", "BID", "OFFER", "MARKET_STATE"],
)


def on_update(item):
    """
    Everytime the subscription get new data, this is run

    1. Load in IG price stream
    2. Resample the raw stream of subscription.
       Need to get it in the form needed for predictions. 1 row per min
    3. Everytime the stream gets larger, make a prediction.
    4. If the prediction is greater than the threshold, BUY and log data of when to sell
    5. Check which dealIds need selling

    """
    if item["values"]["MARKET_STATE"] != "TRADEABLE":
        raise Exception(
            f"Market not open for trading. Market state: {item['values']['MARKET_STATE']}"
        )

    # TODO: @citrez #4  Change raw_stream name to price_stream
    # TODO: #3 Change stream to price_stream_resampled

    append_with_header(prices_stream_responce(item), "raw_stream.csv")
    raw_stream = read_stream("raw_stream.csv")
    # We are resampling everytime time, this is inefficient
    # Dont take the last one since it is not complete yet.
    stream = (
        raw_stream.resample(pd.Timedelta(seconds=60), label="right")
        .last()
        .dropna()  # since if there is a gap in raw stream all the intermediate resamples are filled with nas
        .iloc[:-1, :]
    )
    stream.to_csv(TMP_DIR / "stream.csv", mode="w", header=True)
    with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
        stream.to_sql(name="stream", con=sqliteConnection, if_exists="append")
    stream_length = stream.shape[0]

    # We can only make prediction when there is a stream
    # We only want to make a new prediction when there is a new piece of stream data
    if (stream_length > stream_length_needed) and (
        read_stream_length() < stream_length
    ):
        # When a new row is added to stream_ we jump into action.
        # These things need doing
        # 1. Update the stream length number, so we know when next to jump into action
        # 2. Make a new prediction
        # 3. Check if new prediction warrents buying and buy
        # 4. record information about the buy
        # 4. Check if now we need to sell anything

        # latest_updated_at = stream.tail(1).index.values[0]
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
                **open_position_config_(epic=epic)
            )
            # resp in columns in confirms.create_open_positiion

            logging.info(
                f" { open_position_responce['dealStatus'] } with dealId {open_position_responce['dealId']}"
            )
            # 4
            position_metrics = pd.DataFrame(
                {
                    # "dealreference": [resp["dealReference"]],
                    "dealId": [open_position_responce["dealId"]],
                    "y_pred": [latest_prediction],
                    "model_used": [f"{model_name}-v{model_version}"],
                    "buy_date": [pd.to_datetime(open_position_responce["date"])],
                    "buy_level_resp": [open_position_responce["level"]]
                    # We get this from transactions, but doulbe check
                }
            )
            to_sell = pd.DataFrame(
                {
                    # "dealreference": [resp["dealReference"]],
                    "dealId": [open_position_responce["dealId"]],
                    "buy_date": [pd.to_datetime(open_position_responce["date"])],
                    "to_sell_date": [
                        (
                            pd.to_datetime(open_position_responce["date"]).round("1min")
                            + timedelta(minutes=close_after_x_mins)
                        )
                    ],
                    "sold": False,
                }
            )
            append_with_header(to_sell, "to_sell.csv")
            append_with_header(position_metrics, "position_metrics.csv")
            with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
                position_metrics.to_sql(
                    name="position_metrics", con=sqliteConnection, if_exists="append"
                )
            # append responce
            # This info is in activity??
            single_responce = pd.Series(open_position_responce).to_frame().transpose()
        # 5
        to_sell = pd.read_csv(TMP_DIR / "to_sell.csv")
        current_time = datetime.now()

        need_to_sell_bool = pd.to_datetime(to_sell.to_sell_date) < current_time
        sell_bool = need_to_sell_bool & (to_sell.sold == False)
        logging.info(f"Time now is: {current_time}")
        for i in to_sell[sell_bool].dealId:
            logging.info(f"Closing a position {i}")
            open_position_responce = ig_service.close_open_position(
                **close_position_config_(dealId=i)
            )
            sold = pd.DataFrame(
                # We need this to get the closing refernce for the IG.transactions table
                {
                    "dealId": [i],
                    # "dealreference": [resp["dealReference"]],  # closing reference
                    "dealId": [open_position_responce["dealId"]],  # closing dealId
                    "close_level_resp": open_position_responce[
                        "level"
                    ],  # These should come from IG.transactions, but just checking
                    "profit_resp": open_position_responce[
                        "profit"
                    ],  # These should come from IG.transactions, but just checking
                }
            )
            append_with_header(sold, "sold.csv")
            with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
                sold.to_sql(name="sold", con=sqliteConnection, if_exists="append")

        # update
        to_sell.sold = need_to_sell_bool  # We assume that those that we needed to sell have succesfully been sold
        to_sell.to_csv(TMP_DIR / "to_sell.csv", mode="w", header=True, index=False)

        with sqlite3.connect(TMP_DIR / "autoIG.sqlite") as sqliteConnection:
            # Maybe open and close connection only once at the begining and end?
            # Or wrap everthing in a context manager
            to_sell.to_sql(name="to_sell", con=sqliteConnection, if_exists="append")

        secs_since_deployment = int(
            (current_time - deployment_start ).total_seconds()
        )

        write_to_transations_joined(secs_ago=secs_since_deployment)

    return None


def run():
    sub.addlistener(on_update)
    ig_stream_service.ls_client.subscribe(sub)
    while True:
        user_input = input("Enter dd to termiate: ")
        if user_input == "dd":
            break
    ig_stream_service.disconnect()
    # pd.DataFrame().to_csv(TMP_DIR / "stream_.csv", header=False)  # whipe the tmp data
    # sell all open positions to clean up?


if __name__ == "__main__":
    run()
