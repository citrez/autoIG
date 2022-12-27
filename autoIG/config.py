from enum import Enum
from trading_ig.config import config


def open_position_config_(epic, size=1.0) -> dict[str,str]:
    """The configutation to create an open position"""
    return {
        "epic": epic,
        "currency_code": "GBP",
        "direction": "BUY",
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


def close_position_config_(dealId=None, epic=None,direction = 'SELL',size = 1.0) -> dict[str,str]:
    """
    The configutation to create a closed position.
    When an open position is open, a responce is returned with a dealId.
    This dealId is used to close the position.

    NOTE: Either enter an epic or a deal_id, not both!! (Better to enter a deal_id, more specific. Want to have more fine grained control than closing all epics)
    """
    return {
        "deal_id": dealId,
        "epic": epic,
        "direction": direction,
        "expiry": "DFB",
        "level": None,
        "order_type": "MARKET",
        "quote_id": None,
        "size": size,
    }


ig_service_config = {
    "username": config.username,
    "password": config.password,
    "api_key": config.api_key,
    "acc_type": config.acc_type,
    "acc_number": config.acc_number,
}


class Source(Enum):
    IG = "IG"
    yahoo_finance = "YF"
