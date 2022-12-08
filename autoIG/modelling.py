import pandas as pd
import numpy as np
##Tools to create the TARGET 'r'

def create_future_bid_Open(df, num=3):
    "Add the future periods selling price to the df"
    df_ = df.copy()
    for i in range(1, num + 1):
        df_["BID_OPEN_S" + str(i)] = df_["BID_OPEN"].shift(
            -i
        )  # Next period's bid price is what we can sell it at
    return df_

def generate_target_1(df, goes_up_by, number_of_periods=None) -> pd.Series:
    """
    Strategy for creating returns:
    This sees the number of points it goes above what i bought it for in the next 3 periods.
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

## For transformation steps

def create_past_ask_Open(df, num=3):
    "Add the past periods buying price to the df"
    df_ = df.copy()
    for i in range(1, num + 1):
        df_["ASK_OPEN_S" + str(i)] = df_["ASK_OPEN"].shift(
            i
        )  # Next period's ask price is what we can sell it at
    return df_

def fillna_(df):
    return df.fillna(axis =1,method ='ffill')

def normalise_(df):
    return df / df[['ASK_OPEN']].reindex_like(df).fillna(method='ffill',axis = 'columns')
    
