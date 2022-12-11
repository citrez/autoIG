import pandas as pd
import numpy as np
import joblib
from autoIG.epics import Epics,Tickers
from autoIG.config import Source
from autoIG.utils import print_shape,ROOT_DIR
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import StackingRegressor
from sklearn import set_config
from pathlib import Path
from autoIG.modelling import (
    create_future_bid_Open,
    fillna_,
    create_past_ask_Open,
    normalise_,
    generate_target_2,
    adapt_IG_data_for_training,
    adapt_YF_data_for_training,
)
set_config(transform_output="pandas")
SOURCE = Source["YF"].name
RELOAD_DATA = False
SAVE_MODEL = False
model_config = dict()
model_config["NUMBER_OF_PAST_ASKS"] = 15  # This is for training.
model_config["EPIC"] = Epics.BITCOIN_EPIC.name
model_config["TICKER"] = Tickers.BITCOIN_TICKER.name
historical_prices_config = dict()
historical_prices_config["resolution"] = "1Min"
historical_prices_config["numpoints"] = 500

if RELOAD_DATA:
    if SOURCE == "IG":
        from trading_ig.config import config
        from trading_ig.rest import IGService
        ig_service = IGService(config.username, config.password, config.api_key)
        ig = ig_service.create_session()
        results_ = ig_service.fetch_historical_prices_by_epic(
            model_config["EPIC"], **historical_prices_config
        )
        model_data = results_["prices"]
        model_data.to_pickle("model_data_ig.pkl")
    if SOURCE == "YF":
        import yfinance as yf
        ticker = yf.Ticker(model_config["TICKER"])
        model_data = ticker.history(
            interval="1m", start="2022-12-05", end="2022-12-10"
        )
        model_data.to_pickle("model_data_yf.pkl")
    else:
        Exception("Please provide source to reload data from: (IG/YF)")

else:
    if SOURCE == "IG":
        model_data = pd.read_pickle(Path(__file__).parent/"model_data_ig.pkl")
    if SOURCE == "YF":
        model_data = pd.read_pickle(Path(__file__).parent/"model_data_yf.pkl")

if SOURCE == 'IG':
    model_data= model_data.pipe(adapt_IG_data_for_training)
if SOURCE == 'YF':
    model_data=  model_data.pipe(adapt_YF_data_for_training)
model_data.pipe(print_shape)

create_past_ask_Open_num_small = partial(create_past_ask_Open,num = 3)
create_past_ask_Open_num_medium = partial(create_past_ask_Open,num = 10)
create_past_ask_Open_num_large = partial(create_past_ask_Open,num = 15)

fillna_transformer = FunctionTransformer(fillna_)
normalise_transformer = FunctionTransformer(normalise_)
pl1 = Pipeline(
    [
        ("add_past_period_columns", FunctionTransformer(create_past_ask_Open_num_small)),
        ("fill_na", fillna_transformer),
        ("normalise", normalise_transformer),
        ("predictor", LinearRegression()),
    ]
)
pl2 = Pipeline(
    [
        ("add_past_period_columns", FunctionTransformer(create_past_ask_Open_num_medium)),
        ("fill_na", fillna_transformer),
        ("normalise", normalise_transformer),
        ("predictor", LinearRegression()),
    ]
)
pl3 = Pipeline(
    [
        ("add_past_period_columns", FunctionTransformer(create_past_ask_Open_num_large)),
        ("fill_na", fillna_transformer),
        ("normalise", normalise_transformer),
        ("predictor", LinearRegression()),
    ]
)

stack = StackingRegressor(
    [("small_lookback", pl1), ("medium_lookback", pl2), ("large_lookback", pl3)], final_estimator=LinearRegression()
)

X = model_data[['ASK_OPEN']]
y = model_data['r']
stack.fit(X,y)

if SAVE_MODEL:
    joblib.dump(stack,'model.pkl')
# else:
#     stack = joblib.load(Path(__file__).parent/'model.pkl')



