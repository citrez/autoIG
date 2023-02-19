import pandas as pd
import numpy as np
from autoIG.utils import log_shape

## Tools to create the TARGET 'r'

# Ask is what I buy for
# Bid is what I sell for


def create_future_bid_Open(df, future_periods=1):
    """
    Add the future periods selling price to the df.
    This is important for training, to calculate the profits I would make.
    The target is what I would be able to sell for in one period
    divided by what I can buy for now.
    """
    df_ = df.copy()
    for i in range(1, future_periods + 1):
        # Next period's bid price is what we can sell it at
        # F for plus, P for past
        df_["BID_OPEN_F" + str(i)] = df_["BID_OPEN"].shift(i)
    return df_


def generate_target(df: pd.DataFrame, target_periods_in_future=1) -> pd.DataFrame:
    """
    The simplest target imaginable. How much it has gone up after n (=1) number of periods.
    Just trying to predict the level period, i.e 1 min after
    """
    d = df.copy()
    if target_periods_in_future == 1:
        d["r"] = d["BID_OPEN_F1"] / d["ASK_OPEN"]
    return d  # What I sell for next period / What I buy for this period


## For transformation steps
def create_past_ask_Open(df: pd.DataFrame, past_periods=3):
    "Add the past periods buying price to the df"
    df_ = df.copy()
    for i in range(1, past_periods + 1):
        df_["ASK_OPEN_P" + str(i)] = df_["ASK_OPEN"].shift(
            -i
        )  # Next period's ask price is what we can sell it at
    return df_


def fillna_(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(axis=1, method="ffill")


def normalise_(df):
    """
    Normalises (within row) from the current asking price being 1,
    and all past asking prices normalised
    """
    return df / df[["ASK_OPEN"]].reindex_like(df).fillna(method="ffill", axis="columns")


def adapt_YF_data_for_training(df):
    """
    Yahoo finance data can be used for training.
    We adapt, to get in the same form as used in streaming. (Even though we stream from IG)
    so that the data used for training is consistent
    """
    d = df.copy()
    d = d.set_index("datetime")
    d.index.name = "UPDATED_AT"
    d = d[["open"]].rename(columns={"open": "ASK_OPEN"})
    d["BID_OPEN"] = (
        d["ASK_OPEN"] - 3
    )  # HACK: This data doesnt have bid/ask spread,so I just estimate
    return d[["BID_OPEN", "ASK_OPEN"]]


def adapt_IG_data_for_training(df):
    """
    This takes in the historical data used for training and makes it consistent (column name wise etc).
    With the form of the data being predicted on in production.
    None: This doesnt actually do any of the pre-propcessing steps,
    this is reserved to the pipeline. However, for the pipeline to take place
    it needs to be in the right form.
    Furthermore, the creation of the target it something only done in training
    and therefor is not part of any preprocessing step.
    """
    d = df.copy()
    d.columns = d.columns.get_level_values(0) + "_" + d.columns.get_level_values(1)
    d = d[["ask_Open", "bid_Open"]]
    d = d.rename(columns={"ask_Open": "ASK_OPEN", "bid_Open": "BID_OPEN"})
    d.index.name = "UPDATED_AT"
    return d


## Depreciated
def generate_target_1(df, goes_up_by, number_of_periods=None) -> pd.Series:
    """
    Strategy for creating returns:
    This sees the number of points it goes above what I bought it for in the next 3 periods.
    Note: Here we arbitrarily pick max, we could look at other summary metrics in the next X periods

    TODO: Make our own custom target that incorperates volatility.
    Maybe r = mean(over 3 periods/ over next 1 min) * sd(over 3 periods/ over next 1 min).
    And then in prod we sell after 3 periods
    """
    condlist = [
        (df["BID_OPEN_S1"] - df["ASK_OPEN"]).abs() > goes_up_by,
        (df["BID_OPEN_S2"] - df["ASK_OPEN"]).abs() > goes_up_by,
        (df["BID_OPEN_S3"] - df["ASK_OPEN"]).abs() > goes_up_by,
    ]
    choicelist = [
        np.sign(df["BID_OPEN_S1"] - df["ASK_OPEN"]),
        np.sign(df["BID_OPEN_S2"] - df["ASK_OPEN"]),
        np.sign(df["BID_OPEN_S3"] - df["ASK_OPEN"]),
    ]
    res = np.select(condlist=condlist, choicelist=choicelist, default=0)
    return res


def adapt_yf_data(d_: pd.DataFrame):
    "Adapting data taken from yahoo finance to be suitable to enter the model pipeline"
    d = d_.copy()
    d = (
        d.pipe(adapt_YF_data_for_training)
        .pipe(create_future_bid_Open)
        .pipe(generate_target)
        .dropna()
    )
    d.pipe(log_shape)


# def adapt_ig_data(d_: pd.DataFrame):
#     d = d_.copy()
#     d = (
#             d.pipe(adapt_IG_data_for_training)
#             .pipe(create_future_bid_Open, future_periods=target_periods_in_future)
#             .pipe(generate_target, target_periods_in_future=target_periods_in_future)
#             .dropna()
#         )  # we need this to create the target
#     d.pipe(print_shape)
#     return d
