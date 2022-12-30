from autoIG.instruments import Epics
from datetime import datetime
from autoIG.utils import (
    load_model,
    item_to_df,
    read_stream,
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
# from autoIG.utils import selling_lengths_read_, selling_lengths_write_
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

autoIG_config = dict()
autoIG_config["r_threshold"] = 0.9
autoIG_config["epic"] = Epics.US_CRUDE_OIL.value
autoIG_config['close_after_x_mins'] = 3


sub = Subscription(
    mode="MERGE",
    items=["L1:" + autoIG_config["epic"]],
    fields=["UPDATE_TIME", "BID", "OFFER", "MARKET_STATE"],
)


def on_update(item):
    """
    Everytime the subscription get new data, this is run

    1. Load in IG price stream -
    2. Process the raw stream of subscription.
       Need to get it in the form needed for predictions. 1 row per min

    """
    if item['values']["MARKET_STATE"] != "TRADEABLE":
        print(f"Market state: {item['values']['MARKET_STATE']}")
        raise Exception("Market not open for trading")

    # TODO: Change raw_stream to price_stream
    # TODO: Change stream to price_stream_resampled
    def append_with_header(df,file):
        if os.path.getsize(TMP_DIR / file) == 0:
            df.to_csv(
                TMP_DIR / file, mode="a+", header=True, index=False
            )
        else:
            df.to_csv(
                TMP_DIR / file, mode="a+", header=False, index=False
            )



    def save_to_raw_stream(item):
        # Add headers at the begining, if not don't
        # TODO: This doesnt need to be a dataframe, could just add a line to file.
        append_with_header(item_to_df(item),"raw_stream.csv")
        # if os.path.getsize(TMP_DIR / "raw_stream.csv") == 0:
        #     item_to_df(item).to_csv(
        #         TMP_DIR / "raw_stream.csv", mode="a+", header=True, index=False
        #     )
        # else:
        #     item_to_df(item).to_csv(
        #         TMP_DIR / "raw_stream.csv", mode="a+", header=False, index=False
        #     )

    # save_to_raw_stream(item=item)
    append_with_header(item_to_df(item),"raw_stream.csv")
    raw_stream = read_stream("raw_stream.csv")
    # raw_stream_length = raw_stream_.shape[0]
    # We are resampling everytime time, this is inefficient
    # Dont take the last one since it is not complete yet.
    stream = raw_stream.resample(pd.Timedelta(seconds=60)).last().iloc[:-1, :]
    stream.to_csv(TMP_DIR / "stream.csv", mode="w", header=True)
    stream_length = stream.shape[0]

    # We can only make prediction when there is a stream
    # We only want to make a new prediction when there is a new piece of stream data
    if (stream_length > 0) and (read_stream_length() < stream_length):
        # When a new row is added to stream_ we jump into action.
        # These things need doing
        # 1. Update the stream length number, so we know when next to jump into action
        # 2. Make a new prediction
        # 3. Check if new prediction warrents buying and buy
        # 4. record information about the buy
        # 4. Check if now we need to sell anything
        
        # latest_updated_at = stream.tail(1).index.values[0]
        #1
        write_stream_length(stream_length)

        #2
        predictions = model.predict(stream[["ASK_OPEN"]])  # predict on all stream
        latest_prediction = predictions[-1]
        logging.info(f"Latest prediction: {latest_prediction}")

        #3
        if latest_prediction > autoIG_config["r_threshold"]:
            logging.info("BUY!")
            resp = ig_service.create_open_position(
                **open_position_config_(epic=autoIG_config["epic"])
            )

            logging.info(
                f" { resp['dealStatus'] } with dealId {resp['dealId']}"
            )
            #4
            position_metrics = pd.DataFrame(
                {
                    'dealreference':[resp['dealReference']],
                    "dealId": [resp["dealId"]],
                    "prediction": [latest_prediction],
                    "model_used": [MODEL_PATH],
                    'buy_date':[pd.to_datetime(resp['date'])],
                }
            )
            to_sell = pd.DataFrame(
                {
                    'dealreference':[resp['dealReference']],
                    "dealId": [resp["dealId"]],
                    "to_sell_date": [(pd.to_datetime(resp['date']).round("1min") + timedelta(minutes=autoIG_config['close_after_x_mins']))],
                    'sold' : False
                }
                )
            append_with_header(to_sell, "to_sell.csv")
            append_with_header(position_metrics, "position_metrics.csv")
            # position_metrics.to_csv(
            #     TMP_DIR / "position_metrics.csv", mode="a+", header=False, index=False
            # )

            # append responce
            # This info is in activity??
            single_responce = pd.Series(resp).to_frame().transpose()
            # single_responce.columns = [
            #     "date",
            #     "status",
            #     "reason",
            #     "dealStatus",
            #     "epic",
            #     "expiry",
            #     "dealReference",
            #     "dealId",
            #     "affectedDeals",
            #     "level",
            #     "size",
            #     "direction",
            #     "stopLevel",
            #     "limitLevel",  # levels are for orders
            #     "stopDistance",  # think distance is for deals
            #     "limitDistance",
            #     "guaranteedStop",
            #     "trailingStop",
            #     "profit",
            #     "profitCurrency",
            # ]
            # single_responce = single_responce.set_index("date")  # Date
            # single_responce["date"] = pd.to_datetime(single_responce.date)
            # single_responce["SELL_DATE"] = single_responce.date.round(
            #     "1min"
            # ) + timedelta(minutes=3)
            # single_responce["IS_SOLD"] = False
            # single_responce["MODEL_USED"] = MODEL_PATH
            # single_responce["PREDICTION"] = latest_prediction
            # used_cols = [
            #     "date",  # its the index
            #     "epic",
            #     "dealId",
            #     "dealReference",
            #     "status",
            #     "reason",
            #     "dealStatus",
            #     "direction",
            #     "expiry",
            #     "size",
            #     "stopDistance",
            #     "limitDistance",
            #     "SELL_DATE",
            #     "MODEL_USED",
            #     "PREDICTION",
            # ]
            # single_responce = single_responce[used_cols]
            # if os.path.getsize(TMP_DIR / "responce_.csv") == 0:

            #     single_responce.to_csv(
            #         TMP_DIR / "responce_.csv", mode="a+", header=True, index=False
            #     )
            # else:
            #     single_responce.to_csv(
            #         TMP_DIR / "responce_.csv", mode="a+", header=False, index=False
            #     )
        # Sell positions
        # TODO: check that responce is populated otherwise will get error
        # responce_ = read_responce_()
        # dealIds_to_sell = single_responce[
        #     single_responce.SELL_DATE == latest_updated_at
        # ].dealId

        # for i in dealIds_to_sell:
        #     print(i)
        #     logging.info("Closing a position")

        #     resp = ig_service.close_open_position(
        #         **close_position_config_(dealId=i, epic=autoIG_config["epic"])
        #     )
        #5
        to_sell = pd.read_csv(TMP_DIR/'to_sell.csv')
        current_time =datetime.now()

        need_to_sell_bool =(pd.to_datetime(to_sell.to_sell_date) < current_time)
        sell_bool = need_to_sell_bool  & (to_sell.sold ==False)
        logging.info(f"Time now is: {datetime.now()}")
        for i in to_sell[sell_bool].dealId:
            logging.info(f"Closing a position {i}")
            resp = ig_service.close_open_position(
                **close_position_config_(dealId=i)
            )
            sold = pd.DataFrame(
                {
                    'dealId':[i],
                    'dealreference':[resp['dealReference']], # closing reference
                    "dealId": [resp["dealId"]],
                }
                )
            append_with_header(sold, "sold.csv")
        # update
        to_sell.sold = need_to_sell_bool
        to_sell.to_csv(
                TMP_DIR /'to_sell.csv', mode="w", header=True, index=False
            )
    return None


if __name__ == "__main__":
    sub.addlistener(on_update)
    ig_stream_service.ls_client.subscribe(sub)
    while True:
        user_input = input('Enter dd to termiate')
        if user_input == 'dd':
            break
    ig_stream_service.disconnect()
    # pd.DataFrame().to_csv(TMP_DIR / "stream_.csv", header=False)  # whipe the tmp data
    selling_lengths = list()
    # sell all open positions to clean up 
