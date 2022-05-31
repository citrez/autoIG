
from datetime import datetime
def format_date(d: datetime ):
    return d.strftime("%Y-%m-%d")


def standardise_column_names(df):
    df_new = df.copy()
    df_new.columns = ["_".join(i).lower() for i in df_new.columns]
    return df_new


def to_hours(td):
    "Gives the number of hours diffreence between two timedeltas"
    return td.days * 24 + td.seconds // 3600