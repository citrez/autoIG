"""
Historical price data points per week: 10,000 (Applies to price history endpoints)
"""
import pandas as pd
from pathlib import Path
from trading_ig.rest import IGService
from autoIG.config import ig_service_config
from autoIG.instruments import Epics,Tickers
from autoIG.config import Source
from autoIG.utils import DATA_DIR
import logging

reload_data_config = dict()
reload_data_config["resolution"] = "1min"
epic = Epics.US_CRUDE_OIL.value
ticker = Tickers.US_CRUDE_OIL.value
source = Source.yahoo_finance.value
instrument = epic if source=='IG' else ticker


def fetch_all_training_data():
    # ig_yf_resolution = {"1min": "1m"}
    # resolutions = {"1min": {"IG": "1min", "YF": "1m"}}
    # reload_data_config["numpoints"] = 500

    start_date = "2023-01-02"
    end_date = "2023-01-03"

    model_data_path = (
        DATA_DIR
        / "training"
        / source
        / f"{instrument.replace('.', '_')}"
        / reload_data_config["resolution"]
        
    )
    if source == "IG":
        ig_service = IGService(**ig_service_config)
        _ = ig_service.create_session()
        results_ = ig_service.fetch_historical_prices_by_epic(
            epic=epic, start_date=start_date, end_date=end_date
        )
        model_data = results_["prices"]
        model_data.columns = ["_".join(i) for i in model_data.columns]
        model_data = (
            model_data.reset_index()
        )  # This must be after the unnesting the mutiindex
        model_data.columns = model_data.columns.str.lower()
        model_data.to_csv(model_data_path/ f"{start_date}_to_{end_date}.csv", header=True, index=False)
        logging.info(f"Successfully saved data to {str(model_data_path)}")

    if source == "YF":
        import yfinance as yf

        tick = yf.Ticker(ticker=ticker)
        start = "2022-12-24"
        end = "2022-12-31"  # only 7 days worth of 1m granulairty allow. TODO: Check for this
        model_data = tick.history(interval="1m", start=start, end=end)
        model_data = (
            model_data.reset_index()
        )  # This must be after the unnesting the mutiindex
        model_data.columns = model_data.columns.str.lower()
        model_data_path.mkdir(parents=True,exist_ok = True)
        model_data.to_csv(model_data_path /  f"{start_date}_to_{end_date}.csv",header=  True, index = False)
    else:
        Exception("Please provide source to reload data from: (IG/YF)")
    return None


def make_training_data():
    model_data_path = (
        DATA_DIR
        / "training"
        / "IG"
        / f"{epic.replace('.','_')}"
        / reload_data_config["resolution"]
    )
    if (model_data_path / "full_data.csv").exists():
        (model_data_path / "full_data.csv").unlink()
        logging.info("Removeed full_data.csv")

    dfs = []
    for i in model_data_path.iterdir():
        dfs.append(pd.read_csv(i))
    full_data = pd.concat(dfs, axis="rows")
    full_data.to_csv(model_data_path / "full_data.csv", header=True, index=False)
    logging.info("Created full_data.csv")

    return None





if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(module)-20s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fetch_all_training_data()
