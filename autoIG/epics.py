from enum import Enum

class Epics(Enum):
    "Epics for IG"
    GOLD_EPIC = "MT.D.GC.Month2.IP"
    SANDP_EPIC = "IX.D.SPTRD.DAILY.IP"
    BITCOIN_EPIC = "CS.D.BITCOIN.TODAY.IP"
    US_CRUDE_OIL_EPIC = "CC.D.CL.USS.IP"

class Tickers(Enum):
    "Tickers for yahoo finance"
    GOLD_TICKER = "GC=F"
    MSFT_TICKER = "msft"
    BITCOIN_TICKER = 'BTC-USD'
