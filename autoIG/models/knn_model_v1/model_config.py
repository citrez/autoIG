from autoIG.config import Source
from autoIG.instruments import Epics,Tickers
source = Source.yahoo_finance.value
epic = Epics.US_CRUDE_OIL.value  # Could get both of these based on US_CRUDE_OIL and source
ticker = Tickers.US_CRUDE_OIL.value
threshold = 1.01 # This is the threshold that we should use in prod? But really this should be a seperate piece of analysis?