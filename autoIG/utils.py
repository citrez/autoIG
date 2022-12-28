import datetime
from IPython.display import display
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging


ROOT_DIR = Path(__file__).parent
TMP_DIR = ROOT_DIR / "resources" / "tmp" # This is in the package


def market_series(m) -> tuple[pd.Series]:
    """
    We get information about the market using fetch_market_by_epic method.
    Return the instrument, dealing rules and snapshot series, in that order
    """
    i = pd.Series(m.instrument)
    d = pd.Series(m.dealingRules)
    s = pd.Series(m.snapshot)
    return (i, d, s)


## READ/WRITE I/O
def read_stream(file="stream_.csv", nrows=0):
    path = TMP_DIR / file

    "Read the persistent stream data and take the last 3 rows"
    df = pd.read_csv(
        path,
        # parse_dates=[3],  # after the index has been made
        # dtype={"ASK_OPEN": np.float64, "BID_OPEN": np.float64, "MARKET_STATE": str},
    )
    df = df.set_index("UPDATED_AT")
    df.index = pd.to_datetime(df.index)
    return df[nrows:]


def selling_lengths_read_():
    with open(TMP_DIR / "selling_lengths.csv", "r") as f:
        selling_lengths = f.read().split("\n")[1:]  # first row is empty ''
    # if selling_lengths == ['']:
    #     return list()
    return [int(i) for i in selling_lengths]


def selling_lengths_write_(num):
    with open(TMP_DIR / "selling_lengths.csv", "a") as f:
        f.write("\n" + str(num))


def read_stream_length():
    with open(TMP_DIR / "stream_length.txt", "r") as f:
        l = int(f.read())
    return l


def write_stream_length(n):
    """This is needed to only write to stream when it is bigger than before"""
    with open(TMP_DIR / "stream_length.txt", "w") as f:
        f.write(str(n))
        logging.info(f"New length of stream: {n}")


def read_responce_(file=TMP_DIR / "responce.csv"):
    "Read the persistent responce data"
    df = pd.read_csv(
        file,index_col=0,
    )
    df.index = pd.to_datetime(df.index)
    return df


def load_model(path: str):
    full_path = ROOT_DIR / "resources" / "models" / path
    return joblib.load(full_path)


def parse_time(time: str):
    """
    Add todays date and turn into datetime object.
    Time must be in this format: '16:32:16'
    """
    return datetime.datetime.strptime(
        str(datetime.datetime.now().date()) + " " + time, "%Y-%m-%d %H:%M:%S"
    )


def item_to_df(item) -> pd.DataFrame:
    "Take in a JSON object and parse as df"
    df = pd.DataFrame(
        {
            "UPDATED_AT": [parse_time(item["values"]["UPDATE_TIME"])],
            "BID_OPEN": item["values"]["BID"],
            "ASK_OPEN": item["values"]["OFFER"],
            "MARKET_STATE": item["values"]["MARKET_STATE"],
        }
    )

    df["UPDATED_AT_REAL"] = datetime.datetime.now()

    return df


def format_date(d: datetime.datetime):
    return d.strftime("%Y-%m-%d")


def standardise_column_names(df: pd.DataFrame):
    df_new = df.copy()
    df_new.columns = ["_".join(i).lower() for i in df_new.columns]
    return df_new


def to_hours(td):
    "Gives the number of hours diffreence between two timedeltas"
    return td.days * 24 + td.seconds // 3600


def display_df(df, n=1):
    display(df.head(n))
    return df


def print_shape(df):
    print(f"Shape: {df.shape[0]:,} {df.shape[1]:,}")
    return df
