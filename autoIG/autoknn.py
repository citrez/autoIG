from autoIG.utils import format_date
import pandas as pd
import numpy as np
from autoIG import tickers
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pickle


INTERVAL  = "1d"
def transform_to_knn(df,n = 3):
    s = df.Open.to_list() +[np.NaN] # so that if there arent 4 (at the end), we get an Na
    d = df.Date.to_list()
    col_names = ["m"+str(n-i) for i in range(n+1)]+['id']
    df_dict = [dict(zip( col_names, s[i : (i + n+1) ]+[str(d[i].date())+" : "+str(d[i +n-1].date())]) ) for i in range(len(s) - (n-1)-1 )] # minus an extra1 to account for the na we added on
    df_transformed = pd.DataFrame(
        df_dict
    )

    df_transformed= df_transformed.fillna(np.NaN)
    for i in col_names[-2::-1]:
        df_transformed[i] =  df_transformed[i]/df_transformed[col_names[0]] # get the first and further back col name 

    return df_transformed


history_config_new = {
    "start": format_date(datetime.datetime.now() - relativedelta(days=3)),
    "interval": INTERVAL,
}
gold = yf.Ticker(tickers.GOLD_TICKER)

modern_history = gold.history(**history_config_new).reset_index().reset_index()

filename =  '/Users/ezracitron/my_projects/autoIG/knn_regressor.sav'
# pickle.dump(knn_regressor, open(filename, 'wb'))
model = pickle.load(open(filename, 'rb'))

pred = model.predict(transform_to_knn(modern_history).drop(['m0','id'],axis = 1))[0]
print(pred)

# log this stuff instead
if pred>1.05:
    print(f"We buy!")
else:
    print(f"No buy lads")
