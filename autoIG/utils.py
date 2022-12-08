import datetime
from IPython.display import display
import pandas as pd
from pathlib import Path
import joblib
import logging

ROOT_DIR = Path(__file__).parent
stream_path_ =ROOT_DIR / 'resources'/'tmp'/ "stream_.csv"

def market_series(m) -> tuple[pd.Series]:
    "Return the instrument, dealing rules and snapshot series, in that order"
    i = pd.Series(m.instrument)
    d = pd.Series(m.dealingRules)
    s = pd.Series(m.snapshot)
    return (i, d, s)

def read_stream_(file: Path = stream_path_,nrows = -4):
    "Read the persistent stream data and take the last 3 rows"
    return pd.read_csv(file, names=["UPDATED_AT", "BID_OPEN", "ASK_OPEN"], index_col=0)[nrows:]

def parse_time(time:str):
    """
    Add todays date and turn into datetime object.
    Time must be in this format: '16:32:16'
    """
    return datetime.datetime.strptime( str(datetime.datetime.now().date()) +" "+ time,"%Y-%m-%d %H:%M:%S")

def parse_item(item) -> pd.DataFrame:
    "Take in a JSON object and parse as df"
    return pd.DataFrame(
        {"BID": item["values"]["BID"], "OFFER": item["values"]["OFFER"]},
        columns=["BID", "OFFER"],
        index=[parse_time(item["values"]["UPDATE_TIME"])],
    )

def load_model(path: str):
    full_path = ROOT_DIR / "resources" / "models" / path
    logging.info(f'Model name: {full_path.stem}')
    return joblib.load(full_path)


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
