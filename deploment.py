from autoIG.epics import US_CRUDE_OIL_EPIC
from autoIG.utils import load_model, parse_item, read_stream_, stream_path_
import logging
import time
from trading_ig.config import config
from trading_ig import IGService, IGStreamService
from trading_ig.lightstreamer import Subscription
from autoIG.config import open_position_config_
import pandas as pd

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
    filemode='w',
    filename='log.log'
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
sub = Subscription(
    mode="MERGE",
    items=["L1:" + US_CRUDE_OIL_EPIC],
    fields=["UPDATE_TIME", "BID", "OFFER"],
)

config = dict()
config["r_threshold"] = 0.9995

def on_update(item):
    "Everytime the subscription get new data, this is run"
    parse_item(item).to_csv(stream_path_, mode="a+", header=False)
    stream_ = read_stream_(nrows=-3)
    stream_length = stream_.shape[0]
    predictions = model.predict(stream_[["ASK_OPEN"]])
    latest_prediction = predictions[-1]
    selling_lengths = list()
    print(f"Latest prediction: {latest_prediction}")
    if latest_prediction > config["r_threshold"]:
        logging.info("We BUY!")
        resp = ig_service.create_open_position(**open_position_config_(epic = US_CRUDE_OIL_EPIC))
        logging.info(resp["dealStatus"])
        selling_lengths.append(stream_length+3)
    if stream_length in selling_lengths:
        logging.info('Closing a position')
        resp = ig_service.create_open_position(**open_position_config_(epic = US_CRUDE_OIL_EPIC,size=  -1.0))
        logging.info(resp["dealStatus"])

    # stream_.to_csv("stream_.csv", mode="w", header=False)
    return None


if __name__ == "__main__":
    sub.addlistener(on_update)
    ig_stream_service.ls_client.subscribe(sub)
    time.sleep(50)
    ig_stream_service.disconnect()
    pd.DataFrame().to_csv(stream_path_,header=False ) # whipe the tmp data
