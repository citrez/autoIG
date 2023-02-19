"""
This fetches the latest data from yahoo finance 
Historical price data points per week: 10,000 (Applies to price history endpoints)
"""
import logging
from datetime import datetime, timedelta
import pandas as pd
from autoIG.config import Source
from autoIG.instruments import Epics, Tickers
from autoIG.utils import DATA_DIR
import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


fetch_data_config = dict()
fetch_data_config["resolution"] = "1min"
epic = Epics.US_CRUDE_OIL.value
ticker = Tickers.US_CRUDE_OIL.value
source = Source.yahoo_finance.value
instrument = epic if source == "IG" else ticker
resolution = "1min" if source == "IG" else "1m"


class DataDownloaded(Exception):
    "Trying to fetch data that has already been downloaded"
    pass


# TODO: Find out if they are business days (with pandas) before trying to fetch data
def fetch_history_bucket(start_date, end_date, instrument=instrument):
    """
    This gets historical price data from a start date to end date.
    And saved in an appropriately named directory
    """

    model_data_dir = (
        DATA_DIR / "training" / source / f"{instrument.replace('.', '_')}" / resolution
    )
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_data_path = model_data_dir / f"{start_date}_to_{end_date}.csv"
    if model_data_path.exists():
        raise DataDownloaded("This data has already been downloaded")

    if source == "IG":  # Can I get rid of this?
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
        tick = yf.Ticker(ticker=instrument)
        # only 7 days worth of 1m granulairty allowed. TODO: Check for this
        model_data = tick.history(interval=resolution, start=start_date, end=end_date)
        # This must be after the unnesting the mutiindex
        model_data = model_data.reset_index()
        model_data.columns = model_data.columns.str.lower()
        model_data.to_csv(model_data_path, header=True, index=False)

    return None


def make_training_data(path):
    """
    This takes all the data saved in buckets e.g 2022-01-01_to_2022-01-04
    And concatinates them into a single training data dataset.
    """
    model_data_dir = DATA_DIR / "training" / path
    if (model_data_dir / "full_data.csv").exists():
        (model_data_dir / "full_data.csv").unlink()  # delete
        logging.info("Removed full_data.csv")

    dfs = []
    for file in [f for f in model_data_dir.iterdir() if f.suffix == ".csv"]:
        dfs.append(pd.read_csv(file))
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
        f"datetime range: {str(full_data.datetime.min().date())} : {str(full_data.datetime.max().date() )}"
    )

    return None


def inspect_training_data():
    """
    Take a look at the dates we have in out full_data dataset
    and output an image.
    """

    datetimes = pd.read_csv(
        DATA_DIR / "training/YF/CL=F/1m/full_data.csv",
        usecols=["datetime"],
        parse_dates=True,
    )
    datetimes["count"] = 1
    datetimes = datetimes.set_index("datetime")
    datetimes.index = pd.to_datetime(datetimes.index)

    fig, ax = plt.subplots()
    plt_data = datetimes.resample("120min").count()
    plt.plot(plt_data.index, plt_data["count"])
    fig.set_size_inches(w=12, h=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%y"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.savefig(DATA_DIR / "training/YF/CL=F/1m/fig.png")

    return None


def run_training_data(instument=instrument):
    """
    Perform the actions needed to get,
    update and then inspect the training data
    """
    make_training_data(
        "YF/CL=F/1m"
    )  # Do this first, so that our 'max date' we have is accuate

    max_datetime = pd.read_csv(
        DATA_DIR / "training/YF/CL=F/1m/full_data.csv", usecols=["datetime"]
    ).max()

    max_date = pd.to_datetime(max_datetime)[0].date() + timedelta(days=1)
    yesterday = datetime.now().date() - timedelta(days=1)
    logging.info(f"Max date: {max_date} Yesterday: {yesterday}")
    if max_date < yesterday:
        fetch_history_bucket(start_date=str(max_date), end_date=str(yesterday))
        make_training_data("YF/CL=F/1m")
        inspect_training_data()

    else:
        logging.info("No need to fetch new data and re-make full_data.csv")
    return None


def run_fetch_data():
    pass


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_training_data()
