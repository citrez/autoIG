from autoIG.instruments import Epics
from datetime import datetime
from autoIG.utils import (
    # load_model,
    prices_stream_responce,
    read_stream,
    TMP_DIR,
    read_stream_length,
    write_stream_length,
)
from autoIG.create_data import write_to_transations_joined

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
from mlflow.sklearn import load_model

# Create a custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel('INFO')

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",  # filemode='w', filename='log.log'
)

MODEL_PATH = "/Users/ezracitron/my_projects/auto_IG/mlruns/993369624604660845/c804d0505ffb447ba538023346ca2774/artifacts/model"
model = load_model(MODEL_PATH)
write_stream_length(0)

# Set up Subscription
ig_service = IGService(**ig_service_config)
ig_stream_service = IGStreamService(ig_service)
ig_stream_service.create_session()

# autoIG_config is the config needed for deploymen
# the less the better, should be able to pick up a model and run
autoIG_config = dict()
autoIG_config["r_threshold"] = 0.9
autoIG_config[
    "epic"
] = Epics.US_CRUDE_OIL.value  # market only open to trading until 10pm
autoIG_config["close_after_x_mins"] = 3
autoIG_config["stream_length_needed"] = 3
deployment_start = datetime.now()

sub = Subscription(
    mode="MERGE",
    items=["L1:" + autoIG_config["epic"]],
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
        print(f"Market state: {item['values']['MARKET_STATE']}")
        raise Exception("Market not open for trading")

    # TODO @citrez #4  Change raw_stream name to price_stream
    # TODO: #3 Change stream to price_stream_resampled

    append_with_header(prices_stream_responce(item), "raw_stream.csv")
    raw_stream = read_stream("raw_stream.csv")
    # raw_stream_length = raw_stream_.shape[0]
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

    # We can only make prediction when there is a stream
    # We only want to make a new prediction when there is a new piece of stream data
    if (stream_length > autoIG_config["stream_length_needed"]) and (
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
        if latest_prediction > autoIG_config["r_threshold"]:
            logging.info("BUY!")
            resp = ig_service.create_open_position(
                **open_position_config_(epic=autoIG_config["epic"])
            )
            # resp in columns in confirms.create_open_positiion

            logging.info(f" { resp['dealStatus'] } with dealId {resp['dealId']}")
            # 4
            position_metrics = pd.DataFrame(
                {
                    "dealreference": [resp["dealReference"]],
                    "dealId": [resp["dealId"]],
                    "prediction": [latest_prediction],
                    "model_used": [MODEL_PATH],
                    "buy_date": [pd.to_datetime(resp["date"])],
                    "buy_level_resp": [resp["level"]]
                    # We get this from transactions, but doulbe check
                }
            )
            to_sell = pd.DataFrame(
                {
                    # "dealreference": [resp["dealReference"]],
                    "dealId": [resp["dealId"]],
                    "to_sell_date": [
                        (
                            pd.to_datetime(resp["date"]).round("1min")
                            + timedelta(minutes=autoIG_config["close_after_x_mins"])
                        )
                    ],
                    "sold": False,
                }
            )
            append_with_header(to_sell, "to_sell.csv")
            append_with_header(position_metrics, "position_metrics.csv")
            # append responce
            # This info is in activity??
            single_responce = pd.Series(resp).to_frame().transpose()
        # 5
        to_sell = pd.read_csv(TMP_DIR / "to_sell.csv")
        current_time = datetime.now()

        need_to_sell_bool = pd.to_datetime(to_sell.to_sell_date) < current_time
        sell_bool = need_to_sell_bool & (to_sell.sold == False)
        logging.info(f"Time now is: {datetime.now()}")
        for i in to_sell[sell_bool].dealId:
            logging.info(f"Closing a position {i}")
            resp = ig_service.close_open_position(**close_position_config_(dealId=i))
            sold = pd.DataFrame(
                # We need this to get the closing refernce for the IG.transactions table
                {
                    "dealId": [i],
                    "dealreference": [resp["dealReference"]],  # closing reference
                    "dealId": [resp["dealId"]],
                    "close_level_resp": resp[
                        "level"
                    ],  # These should come from IG.transactions, but just checking
                    "profit_resp": resp[
                        "profit"
                    ],  # These should come from IG.transactions, but just checking
                }
            )
            append_with_header(sold, "sold.csv")
        # update
        to_sell.sold = need_to_sell_bool
        to_sell.to_csv(TMP_DIR / "to_sell.csv", mode="w", header=True, index=False)

        mins_since_deployment = int(
            (deployment_start - current_time).total_seconds() / 60
        )

        write_to_transations_joined(secs_ago=mins_since_deployment + 360)

    return "hi"


def run():
    _ = sub.addlistener(on_update)
    print(_)
    _ = ig_stream_service.ls_client.subscribe(sub)
    print(_)
    while True:
        user_input = input("Enter dd to termiate: ")
        if user_input == "dd":
            break
    ig_stream_service.disconnect()
    # pd.DataFrame().to_csv(TMP_DIR / "stream_.csv", header=False)  # whipe the tmp data
    # sell all open positions to clean up?


if __name__ == "__main__":
    run()
