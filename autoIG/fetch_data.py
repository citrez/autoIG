"""
Historical price data points per week: 10,000 (Applies to price history endpoints)
"""
import pandas as pd

from autoIG.instruments import Epics, Tickers
from autoIG.config import Source
from autoIG.utils import DATA_DIR
import logging

from datetime import timedelta, datetime


reload_data_config = dict()
reload_data_config["resolution"] = "1min"
epic = Epics.US_CRUDE_OIL.value
ticker = Tickers.US_CRUDE_OIL.value
source = Source.yahoo_finance.value
instrument = epic if source == "IG" else ticker
resolution = "1min" if source == "IG" else "1m"

# ig_yf_resolution = {"1min": "1m"}
# resolutions = {"1min": {"IG": "1min", "YF": "1m"}}
# reload_data_config["numpoints"] = 500


class DataDownloaded(Exception):
    "Trying to fetch data that has already been downloaded"
    pass


# TODO: Find out if they are business days (with pandas) before trying to fetch data
def fetch_all_training_data(start_date, end_date):

    model_data_dir = (
        DATA_DIR / "training" / source / f"{instrument.replace('.', '_')}" / resolution
    )
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_data_path = model_data_dir / f"{start_date}_to_{end_date}.csv"
    if model_data_path.exists():
        raise DataDownloaded("This data has already been downloaded")

    if source == "IG":
        from trading_ig.rest import IGService
        from autoIG.config import ig_service_config

        ig_service = IGService(**ig_service_config)
        _ = ig_service.create_session()
        results_ = ig_service.fetch_historical_prices_by_epic(
            epic=instrument, start_date=start_date, end_date=end_date
        )
        model_data = results_["prices"]
        model_data.columns = ["_".join(i) for i in model_data.columns]
        model_data = (
            model_data.reset_index()
        )  # This must be after the unnesting the mutiindex
        model_data.columns = model_data.columns.str.lower()
        model_data.to_csv(model_data_path, header=True, index=False)
        logging.info(f"Successfully saved data to {str(model_data_dir)}")

    if source == "YF":

        import yfinance as yf

        tick = yf.Ticker(ticker=instrument)
        # only 7 days worth of 1m granulairty allowed. TODO: Check for this
        model_data = tick.history(interval=resolution, start=start_date, end=end_date)
        # This must be after the unnesting the mutiindex
        model_data = model_data.reset_index()
        model_data.columns = model_data.columns.str.lower()
        model_data.to_csv(model_data_path, header=True, index=False)

    return None


def make_training_data(path):
    model_data_dir = DATA_DIR / "training" / path
    if (model_data_dir / "full_data.csv").exists():
        (model_data_dir / "full_data.csv").unlink()
        logging.info("Removeed full_data.csv")

    dfs = []
    for i in model_data_dir.iterdir():
        dfs.append(pd.read_csv(i))
    full_data = pd.concat(dfs, axis="rows")
    full_data = full_data.sort_values("datetime", ascending=False)
    full_data["datetime"] = pd.to_datetime(
        full_data["datetime"].str.removesuffix("-05:00")
    )
    model_data_path = model_data_dir / "full_data.csv"
    full_data.to_csv(model_data_path, header=True, index=False)
    logging.info(f"Created full_data.csv")
    logging.info(f"full_data rows: {full_data.shape[0]:,}")
    logging.info(
        f"datetime range: {full_data.datetime.min()} : {full_data.datetime.max()}"
    )

    return None


def inspect_training_data():
    "Take a look at the dates we have and output an image"
    datetimes = pd.read_csv(
        DATA_DIR / "training/YF/CL=F/1m/full_data.csv",
        usecols=["datetime"],
        parse_dates=True,
    )
    datetimes["count"] = 1
    datetimes = datetimes.set_index("datetime")
    datetimes.index = pd.to_datetime(datetimes.index)
    import matplotlib.pyplot as plt

    datetimes.resample("60min").count().plot()
    plt.savefig(DATA_DIR / "training/YF/CL=F/1m/fig.png")
    return None


def training_data_runner():
    "Perform the actions needed to get and update training data"
    max_datetime = pd.read_csv(
        DATA_DIR / "training/YF/CL=F/1m/full_data.csv", usecols=["datetime"]
    ).max()

    max_date = pd.to_datetime(max_datetime)[0].date() + timedelta(days=1)
    yesterday = datetime.now().date() - timedelta(days=1)
    if max_date < yesterday:
        fetch_all_training_data(start_date=str(max_date), end_date=str(yesterday))
        make_training_data("YF/CL=F/1m")
    else:
        logging.info("No need to fetch new data and re-make full_data.csv")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    training_data_runner()
    inspect_training_data()
