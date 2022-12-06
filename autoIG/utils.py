
from datetime import datetime
from IPython.display import display
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__)

def format_date(d: datetime ):
    return d.strftime("%Y-%m-%d")


def standardise_column_names(df: pd.DataFrame):
    df_new = df.copy()
    df_new.columns = ["_".join(i).lower() for i in df_new.columns]
    return df_new


def to_hours(td):
    "Gives the number of hours diffreence between two timedeltas"
    return td.days * 24 + td.seconds // 3600

def display_df(df,n=1):
    display(df.head(n))
    return df

def print_shape(df):
    print(f'Shape: {df.shape[0]:,} {df.shape[1]:,}')
    return df


