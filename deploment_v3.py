from autoIG.instruments import Epics
from autoIG.utils import (
    load_model,
    parse_item,
    read_stream_,
    read_responce_,
    TMP_DIR,
    read_stream_length,
    write_stream_length,
)
import os
import logging
import time
from trading_ig import IGService, IGStreamService
from trading_ig.lightstreamer import Subscription
from autoIG.config import (
    open_position_config_,
    close_position_config_,
    ig_service_config,
)
from autoIG.utils import selling_lengths_read_, selling_lengths_write_
import pandas as pd
from datetime import timedelta

# Create a custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel('INFO')


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    # filemode='w',
    # filename='log.log'
)

MODEL_PATH = "base_model_v3/model.pkl"
model = load_model(MODEL_PATH)
write_stream_length(0)

# Set up Subscription
ig_service = IGService(**ig_service_config)
ig_stream_service = IGStreamService(ig_service)
ig_stream_service.create_session()

autoIG_config = dict()
autoIG_config["r_threshold"] = 0.9
autoIG_config["epic"] = Epics.US_CRUDE_OIL.value

sub = Subscription(
    mode="MERGE",
    items=["L1:" + autoIG_config["epic"]],
    fields=["UPDATE_TIME", "BID", "OFFER", "MARKET_STATE"],
)


def on_update(item):
    """
    Everytime the subscription get new data, this is run

    1. Process the raw stream of subscription.
    Need to get it in the form needed for predictions. 1 row per min

    """
    if os.path.getsize(TMP_DIR / "raw_stream_.csv") == 0:
        parse_item(item).to_csv(
            TMP_DIR / "raw_stream_.csv", mode="a+", header=True, index=False
        )
    else:
        parse_item(item).to_csv(
            TMP_DIR / "raw_stream_.csv", mode="a+", header=False, index=False
        )

    raw_stream_ = read_stream_("raw_stream_.csv")
    raw_stream_length = raw_stream_.shape[0]

    stream_ = (  # We are resampling everytime time, this is inefficient
        raw_stream_.resample(pd.Timedelta(seconds=60)).last().iloc[:-1, :]
    )  # Dont take the last one since it is not complete yet.
    stream_.to_csv(TMP_DIR / "stream_.csv", mode="w", header=True)
    stream_length = stream_.shape[0]

    # We can only make prediction when there is a stream
    # We only want to make a new prediction when there is a new piece of stream data
    if stream_length > 0 and (read_stream_length() < stream_length):
        # When a new row is added to stream_ we jump into action.
        # These things need doing
        # 1. Update the stream length number, so we know when next to jump into action
        # 2. Make a new prediction
        # 3. Check if new prediction warrents buying
        # 4. Check if now we need to sell anything
        latest_updated_at = stream_.tail(1).index.values[0]
        write_stream_length(l=stream_length)

        predictions = model.predict(stream_[["ASK_OPEN"]])  # predict on all stream_
        latest_prediction = predictions[-1]
        print(f"Latest prediction {latest_prediction}")

        if latest_prediction > autoIG_config["r_threshold"]:
            logging.info("We BUY!")
            resp = ig_service.create_open_position(
                **open_position_config_(epic=autoIG_config["epic"])
            )
            logging.info(
                f" { resp['dealStatus'] } becuase { resp['reason'] } with dealId {resp['dealId']}"
            )

            # append responce
            single_responce = pd.Series(resp).to_frame().transpose()
            single_responce.columns = [
                "date",
                "status",
                "reason",
                "dealStatus",
                "epic",
                "expiry",
                "dealReference",
                "dealId",
                "affectedDeals",
                "level",
                "size",
                "direction",
                "stopLevel",
                "limitLevel",  # levels are for orders
                "stopDistance",  # think distance is for deals
                "limitDistance",
                "guaranteedStop",
                "trailingStop",
                "profit",
                "profitCurrency",
            ]
            # single_responce = single_responce.set_index("date")  # Date
            single_responce["date"] = pd.to_datetime(single_responce.date)
            single_responce["SELL_DATE"] = single_responce.date.round(
                "1min"
            ) + timedelta(minutes=3)
            single_responce["IS_SOLD"] = False
            single_responce["MODEL_USED"] = MODEL_PATH
            single_responce["PREDICTION"] = latest_prediction
            used_cols = [
                "date",  # its the index
                "epic",
                "dealId",
                "dealReference",
                "status",
                "reason",
                "dealStatus",
                "direction",
                "expiry",
                "size",
                "stopDistance",
                "limitDistance",
                "SELL_DATE",
                "MODEL_USED",
                "PREDICTION",
            ]
            single_responce = single_responce[used_cols]
            if os.path.getsize(TMP_DIR / "responce_.csv") == 0:

                single_responce.to_csv(
                    TMP_DIR / "responce_.csv", mode="a+", header=True, index=False
                )
            else:
                single_responce.to_csv(
                    TMP_DIR / "responce_.csv", mode="a+", header=False, index=False
                )
        # Sell positions
        # TODO: check that responce is populated otherwise will get error
        responce_ = read_responce_()
        dealIds_to_sell = single_responce[
            single_responce.SELL_DATE == latest_updated_at
        ].dealId

        for i in dealIds_to_sell:
            print(i)
            logging.info("Closing a position")

            resp = ig_service.close_open_position(
                **close_position_config_(dealId=i, epic=autoIG_config["epic"])
            )

    return None


if __name__ == "__main__":
    sub.addlistener(on_update)
    ig_stream_service.ls_client.subscribe(sub)
    time.sleep(300)
    ig_stream_service.disconnect()
    # pd.DataFrame().to_csv(TMP_DIR / "stream_.csv", header=False)  # whipe the tmp data
    selling_lengths = list()
