from datetime import datetime
import pandas as pd


def format_date(d: datetime):
    return d.strftime("%Y-%m-%d")

def market_series(m) -> tuple[pd.Series]:
    """
    We get information about the market using IG's `fetch_market_by_epic` method.
    Return the instrument, dealing rules and snapshot series, in that order
    """
    i = pd.Series(m.instrument)
    d = pd.Series(m.dealingRules)
    s = pd.Series(m.snapshot)
    return (i, d, s)

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
