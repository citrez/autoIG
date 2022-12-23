from enum import Enum

class Epics(Enum):
    "Epics for IG"
    GOLD = "MT.D.GC.Month2.IP"
    SANDP = "IX.D.SPTRD.DAILY.IP"
    BITCOIN = "CS.D.BITCOIN.TODAY.IP"
    US_CRUDE_OIL = "CC.D.CL.USS.IP"

class Tickers(Enum):
    "Tickers for yahoo finance"
    GOLD = "GC=F"
    MSFT = "msft"
    BITCOIN = 'BTC-USD'
    US_CRUDE_OIL = 'CL=F'
