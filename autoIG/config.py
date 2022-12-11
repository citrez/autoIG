from enum import Enum

def open_position_config_(epic,size = 1.0):
    return {
    "currency_code": "GBP",
    "direction": "BUY",
    "epic": epic,
    "order_type": "MARKET",
    "expiry": "DFB",
    "force_open": "false",
    "guaranteed_stop": "false",
    "size": size,
    "level": None,
    "limit_distance": None,
    "limit_level": None,
    "quote_id": None,
    "stop_level": None,
    "stop_distance": None,
    "trailing_stop": None,
    "trailing_stop_increment": None,
}


class Source(Enum):
    IG="IG"
    YF='YF'