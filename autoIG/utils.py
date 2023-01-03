from datetime import datetime
from IPython.display import display
import pandas as pd
from pathlib import Path
import joblib
import logging
import os

# Paths
ROOT_DIR = Path(__file__).parent
TMP_DIR = ROOT_DIR / "resources" / "tmp" # This is in the package TODO: Move outside package

def market_series(m) -> tuple[pd.Series]:
    """
    We get information about the market using IG's `fetch_market_by_epic` method.
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

def read_stream_length():
    with open(TMP_DIR / "stream_length.txt", "r") as f:
        l = int(f.read())
    return l


def write_stream_length(n):
    """This is needed to only write to stream when it is bigger than before"""
    with open(TMP_DIR / "stream_length.txt", "w") as f:
        f.write(str(n))
        logging.info(f"New length of stream: {n}")

# We can get responce from activity IG table
# def read_responce_(file=TMP_DIR / "responce.csv"):
#     "Read the persistent responce data"
#     df = pd.read_csv(
#         file,index_col=0,
#     )
#     df.index = pd.to_datetime(df.index)
#     return df

# DEPRECIATED. Get MLflow to save models automatically
# def load_model(path: str):
#     full_path = ROOT_DIR / "resources" / "models" / path
#     return joblib.load(full_path)

def parse_time(time: str):
    """
    Add todays date and turn into datetime object.
    Time must be in this format: '16:32:16'
    """
    return datetime.strptime(
        str(datetime.now().date()) + " " + time, "%Y-%m-%d %H:%M:%S"
    )

def prices_stream_responce(item) -> pd.DataFrame:
    "Take in a JSON object and parse as df"
    df = pd.DataFrame(
        {
            "UPDATED_AT": [parse_time(item["values"]["UPDATE_TIME"])],
            "BID_OPEN": item["values"]["BID"],
            "ASK_OPEN": item["values"]["OFFER"],
            "MARKET_STATE": item["values"]["MARKET_STATE"],
        }
    )

    # df["UPDATED_AT_REAL"] = datetime.now()
    return df

def append_with_header(df, file):
    if os.path.getsize(TMP_DIR / file) == 0:
        df.to_csv(TMP_DIR / file, mode="a+", header=True, index=False)
    else:
        df.to_csv(TMP_DIR / file, mode="a+", header=False, index=False)



    
def mins_to_ms(m = 1):
    return 1000*60*m

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
