from trading_ig.rest import IGService
from trading_ig.config import config
import numpy as np
from typing import Dict, Any


class Trader:
    def __init__(self, epic,config = config) -> None:

        self.ig_service = IGService(config.username, config.password, config.api_key)
        self.ig = self.ig_service.create_session()
        self.epic = epic
        self.market = self.ig_service.fetch_market_by_epic(self.epic)
        self.minDealSize = self.market.dealingRules["minDealSize"]["value"]

    def get_open_position_totals(self):
        open_positions = self.ig_service.fetch_open_positions()

        open_positions = open_positions.assign(
            direction_signed=lambda df: np.where(df.direction == "SELL", -1, 1),
            size_signed=lambda df: df["size"] * df.direction_signed,
        )
        open_positions_totals = open_positions.groupby("epic", as_index=False)[
            "size_signed"
        ].sum()
        return open_positions_totals

    def create_open_position_config(self,size:float=None, direction:str="BUY")-> Dict[str,Any]:
        """Creates the config need to create a position.
        Uses the instances state to get market expriy information and epic code etc.  

        Args:
            size: The size of the trade
            direction : To buy or to sell
        """
        expiry = self.market.instrument["expiry"]
        res = {
            "currency_code": "GBP",
            "direction": direction,
            "epic": self.epic,
            "order_type": "MARKET",
            "expiry": expiry,
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
        return res