import numpy as np
from trading_ig.config import config
from trading_ig.rest import IGService
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


def format_date(d):
    return d.strftime("%Y-%m-%d")


def standardise_column_names(df):
    df_new = df.copy()
    df_new.columns = ["_".join(i).lower() for i in df_new.columns]
    return df_new


def to_hours(td):
    "Gives the number of hours diffreence between two timedeltas"
    return td.days * 24 + td.seconds // 3600


def get_open_position_totals():
    open_positions = ig_service.fetch_open_positions()

    open_positions = open_positions.assign(
        direction_signed=lambda df: np.where(df.direction == "SELL", -1, 1),
        size_signed=lambda df: df["size"] * df.direction_signed,
    )
    open_positions_totals = open_positions.groupby("epic", as_index=False)[
        "size_signed"
    ].sum()
    return open_positions_totals


# config
ENABLE_TRADING = True
DAYS_AWAY = 10
STARTDATE = format_date(datetime.now() - timedelta(days=DAYS_AWAY))
ENDDATE = format_date(datetime.now() + timedelta(days=1))
RESOLUTION = "H"
GOLD_EPIC = "MT.D.GC.Month2.IP"
SANDP_EPIC = "IX.D.SPTRD.DAILY.IP"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create handlers
file_handler = logging.FileHandler("trade.log")
# Create formatters and add it to handlers
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(f_format)
logger.addHandler(file_handler)

ig_service = IGService(config.username, config.password, config.api_key)
ig = ig_service.create_session()


class Trader:
    def __init__(self, epic) -> None:
        self.epic = epic
        self.market = ig_service.fetch_market_by_epic(self.epic)
        self.minDealSize = self.market.dealingRules["minDealSize"]["value"]


gold_trader = Trader(GOLD_EPIC)


def create_open_position_config(epic, size=None, direction="BUY"):

    market = ig_service.fetch_market_by_epic(epic)
    expiry = gold_trader.market.instrument["expiry"]
    minsize = gold_trader.minDealSize
    if size is None:
        size = minsize
    else:
        if size < minsize:
            # logger.info("No order placed. Size of trade smaller than minimum")
            size = 0
    res = {
        "currency_code": "GBP",
        "direction": direction,
        "epic": epic,
        "order_type": "MARKET",
        "expiry": expiry,
        "force_open": "false",
        "guaranteed_stop": "false",
        "size": size,
        "level": None,
        "limit_distance": None,
        "limit_level": None,
        "quote_id": None,
        "stop_level": None,
        "stop_distance": None,
        "trailing_stop": None,
        "trailing_stop_increment": None,
    }
    return res


if __name__ == "__main__":
    result = ig_service.fetch_historical_prices_by_epic(
        epic=GOLD_EPIC, start_date=STARTDATE, end_date=ENDDATE, resolution=RESOLUTION
    )
    prices_raw = result["prices"]
    prices = standardise_column_names(prices_raw)
    prices = prices.reset_index()
    prices = prices.assign(
        days_since=lambda df: list(
            map(lambda x: to_hours(x), (df["DateTime"] - df["DateTime"][0]))
        )
    )

    X = prices.days_since.to_numpy().reshape(-1, 1)
    y = prices.bid_open

    # build preprocessing pipeline
    linearModpipeline = Pipeline([("poly", PolynomialFeatures(degree=3))])
    # fit the pipeline on this data
    # and the transform this data using that fit
    X_preprocessed = linearModpipeline.fit_transform(X)

    linearMod = LinearRegression(fit_intercept=True)
    linearMod.fit(X_preprocessed, y)

    # predict the next value
    y_predictions = linearMod.predict(
        linearModpipeline.transform((X[-1] + 1).reshape(-1, 1))
    )
    y_predictions
    open_positions_totals = get_open_position_totals()

    downscaling = 1 / 10
    total_size = open_positions_totals[open_positions_totals.epic == GOLD_EPIC][
        "size_signed"
    ]
    total_size = float(total_size)
    predicted_increase = y_predictions[0] - y.iloc[-1]
    predicted_increase_downscaled = predicted_increase * downscaling
    change_needed = (predicted_increase_downscaled - total_size).round(2)
    logger.info(f"Size wanted: {predicted_increase.round(2)}")
    logger.info(f"Size wanted downscaled: {predicted_increase_downscaled.round(2)}")
    logger.info(f"Toal size of current trades: {total_size}")
    logger.info(f"change needed: {change_needed}")

    if ENABLE_TRADING:
        if abs(change_needed) < gold_trader.minDealSize:
            logger.info("Trade not large enough")

        elif change_needed > 0:
            print("We're buying!")
            res = ig_service.create_open_position(
                **create_open_position_config(
                    GOLD_EPIC, size=abs(change_needed), direction="BUY"
                )
            )
            print(res)
        elif change_needed < 0:
            print("We're selling!")
            res = ig_service.create_open_position(
                **create_open_position_config(
                    GOLD_EPIC, direction="SELL", size=abs(change_needed)
                )
            )
            print(res)
        elif change_needed == 0:
            print("No trade needed")
