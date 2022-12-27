import pandas as pd
import numpy as np
##Tools to create the TARGET 'r'

def create_future_bid_Open(df, num=3):
    """Add the future periods selling price to the df.
    This is important for training, to calculate the profits I would make"""
    df_ = df.copy()
    for i in range(1, num + 1):
        df_["BID_OPEN_S" + str(i)] = df_["BID_OPEN"].shift(
            -i
        )  # Next period's bid price is what we can sell it at
    return df_

def generate_target_2(df,number_of_periods=None):
    """Just trying to predict the level period, i.e 1 min after"""
    return df["BID_OPEN_S1"] / df["ASK_OPEN"] # What I sell for next period / What I buy for this period 


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
    "Normalises (within row) from the current asking price being 1, and all past asking prices normalised"
    return df / df[['ASK_OPEN']].reindex_like(df).fillna(method='ffill',axis = 'columns')
    
def adapt_YF_data_for_training(df_):
    """
    Yahoo finance data can be used for training.
    We adapt, to get in the same form as used in streaming
    so that the data used for training is consistent
    """
    df = df_.copy()
    df.index.name = 'UPDATED_AT'
    df = df[['Open']].rename(columns = {'Open':'ASK_OPEN'})
    df['BID_OPEN'] = df['ASK_OPEN'] + 3 # Hack, this data doesnt have bid/ask spread
    df = create_future_bid_Open(df,num = 1) # we need this to create the target
    df['r'] = generate_target_2(df)
    df = df.dropna()
    return df

def adapt_IG_data_for_training(df_):
    """
    This takes in the historical data used for training and makes it consistent (column name wise etc).
    With the form of the data being predicted on in production.
    None: This doesnt actually do any of the pre-propcessing steps, 
    this is reserved to the pipeline. However, for the pipeline to take place 
    it needs to be in the right form.
    Furthermore, the creation of the target it something only done in training 
    and therefor is not part of any preprocessing step.
    """
    df = df_.copy()
    
    df.columns = (
        df.columns.get_level_values(0)
        + "_"
        + df.columns.get_level_values(1)
    )
    df = df[["ask_Open", "bid_Open"]]
    df = df.rename(columns={ "ask_Open": "ASK_OPEN","bid_Open": "BID_OPEN"})
    df.index.name = 'UPDATED_AT'
    df = create_future_bid_Open(df,num = 1) # we need this to create the target
    df['r'] = generate_target_2(df)
    df = df.dropna()
    return df

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
