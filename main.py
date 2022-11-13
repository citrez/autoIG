import numpy as np
from trading_ig.config import config
from trading_ig.rest import IGService
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from autoIG import utils, trade, epics

gold_trader = trade.Trader(epics.GOLD_EPIC)


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
PERIOD_DAYS = 3
STARTDATE = utils.format_date(datetime.now() - timedelta(days=PERIOD_DAYS))
ENDDATE = utils.format_date(datetime.now() + timedelta(days=1))
RESOLUTION = "D"
GOLD_EPIC = "MT.D.GC.Month2.IP"
SANDP_EPIC = "IX.D.SPTRD.DAILY.IP"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename="trade.log")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(f_format)
logger.addHandler(file_handler)

ig_service = IGService(config.username, config.password, config.api_key)
ig = ig_service.create_session()


gold_trader = trade.Trader(GOLD_EPIC)


if __name__ == "__main__":
    logger.debug(f"Getting history, Start date: {STARTDATE}, end data: {ENDDATE}, resolution: {RESOLUTION}")
    result = gold_trader.ig_service.fetch_historical_prices_by_epic(
        epic=gold_trader.epic, start_date=STARTDATE, end_date=ENDDATE, resolution=RESOLUTION
    )
    prices_raw = result["prices"]
    prices = utils.standardise_column_names(prices_raw)
    prices = prices.reset_index()
    prices = prices.assign(
        days_since=lambda df: list(
            map(lambda x: utils.to_hours(x), (df["DateTime"] - df["DateTime"][0]))
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
    logger.info(f"Toal size (sum) of current positions: {total_size}")
    logger.info(f"Size wanted: {predicted_increase.round(2)}")
    logger.info(f"Size wanted downscaled: {predicted_increase_downscaled.round(2)}")
    logger.info(f"change needed: {change_needed}")

    if ENABLE_TRADING:
        if abs(change_needed) > gold_trader.minDealSize:
            if change_needed < 0:
                logger.info("We're selling!")
                direction="SELL"
            if change_needed < 0:
                logger.info("We're buying!")
                direction="BUY"
            res = gold_trader.ig_service.create_open_position(
                **gold_trader.create_open_position_config(
                    size=change_needed, direction=direction
                )
            )
            logger.info(res)
        else:
            logger.info("Trade not large enough")
