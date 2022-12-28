from autoIG.instruments import Epics
from autoIG.utils import (
    load_model,
    item_to_df,
    read_stream,
    TMP_DIR,
    read_stream_length,
    write_stream_length,
)
import logging
import time
from trading_ig.config import config
from trading_ig import IGService, IGStreamService
from trading_ig.lightstreamer import Subscription
from autoIG.config import open_position_config_
from autoIG.utils import selling_lengths_read_, selling_lengths_write_
import pandas as pd

import mlflow
import mlflow.sklearn


def read_responce_(file=TMP_DIR / "responce_.csv"):
    "Read the persistent responce data and take the last 3 rows"
    df = pd.read_csv(
        file,
        names=[
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
        ],
        index_col=0,
    )
    cols = [
        "date",
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
    ]
    df = df[cols]
    df["SELL_DATE"] = df["DATE"]
    df['MODEL_USED'] = 'model'
    df['PREDICTION'] = '0.9984'
    df['']
    return df[cols]


# Create a custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel('INFO')

# Create handlers
# console_handler = logging.StreamHandler(sys.stdout)
# file_handler = logging.FileHandler(filename='log.log',mode='w')

# Create formatters and add it to handlers
# console_format = logging.Formatter('%(message)s')
# file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
# console_handler.setFormatter(console_format)
# file_handler.setFormatter(file_format)
# console_handler.setLevel(level='WARNING')
# file_handler.setLevel(level='INFO')
# Add handlers to the logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    # filemode='w',
    # filename='log.log'
)

model = load_model("base_model_v2/model.pkl")

# Set up Subscription
ig_service = IGService(
    config.username,
    config.password,
    config.api_key,
    config.acc_type,
    acc_number=config.acc_number,
)
ig_stream_service = IGStreamService(ig_service)
ig_stream_service.create_session()
autoIG_config = dict()
autoIG_config["r_threshold"] = 0.9
autoIG_config["epic"] = Epics.BITCOIN_EPIC.value

sub = Subscription(
    mode="MERGE",
    items=["L1:" + autoIG_config["epic"]],
    fields=["UPDATE_TIME", "BID", "OFFER","MARKET_STATE"],
)


def on_update(item):
    "Everytime the subscription get new data, this is run"

    item_to_df(item).to_csv(TMP_DIR / "raw_stream_.csv", mode="a+", header=False)

    raw_stream_length, raw_stream_ = read_stream("raw_stream_.csv")
    # Right now we are resampling everytime time, this is inefficient
    stream_ = (
        raw_stream_.resample(pd.Timedelta(seconds=60)).last().iloc[:-1, :]
    )  # Dont take the last one since it is not complete yet.
    stream_.to_csv(TMP_DIR / "stream_.csv", mode="w", header=True)
    stream_length = stream_.shape[0]
    old_stream_length = read_stream_length()
    if stream_.shape[0] > 0 and (old_stream_length < stream_length):
        write_stream_length(l=stream_length)
        predictions = model.predict(stream_[["ASK_OPEN"]])  # predict on all stream_
        print(predictions)
        latest_prediction = predictions[-1]
        # print(f"Latest prediction: {latest_prediction}")
        if latest_prediction > autoIG_config["r_threshold"]:
            logging.info("We BUY!")
            resp = ig_service.create_open_position(
                **open_position_config_(epic=autoIG_config["epic"])
            )
            logging.info(
                f" { resp['dealStatus'] } becuase { resp['reason'] } with dealId {resp['dealId']}"
            )

            def append_responce_(resp):
                s = pd.Series(resp).to_frame().transpose()

                s.to_csv(
                    TMP_DIR / "responce_.csv", mode="a+", header=False, index=False
                )

            pd.Series(resp).to_frame().transpose().to_csv(
                TMP_DIR / "responce_.csv", mode="a+", header=False, index=False
            )

            selling_lengths_write_(stream_length + 3)
        if stream_length in selling_lengths_read_():
            logging.info("Closing a position")
            # def close_position_config_(dealId):
            #     return {
            #     "dealId": dealId,
            #     "direction": "BUY",
            #     "epic": US_CRUDE_OIL_EPIC,
            #     "expiry": 'DFB',
            #     "level": None,
            #     "order_type": "MARKET",
            #     "quoteId": None,
            #     "size": 1.0,
            # }
            # resp = ig_service.create_open_position(
            #     **close_position_config_()
            # )
            # logging.info(f" { resp['dealStatus'] } becuase { resp['reason'] }")

    return None


if __name__ == "__main__":
    sub.addlistener(on_update)
    ig_stream_service.ls_client.subscribe(sub)
    time.sleep(300)
    ig_stream_service.disconnect()
    # pd.DataFrame().to_csv(TMP_DIR / "stream_.csv", header=False)  # whipe the tmp data
    selling_lengths = list()
