"Ulitimately this should all be data already in the database and this should be a view/ task"
from trading_ig import IGService
from autoIG.config import ig_service_config

ig_service = IGService(**ig_service_config)
import pandas as pd


from datetime import datetime
from datetime import timedelta

from autoIG.utils import TMP_DIR


def write_to_transations_joined(secs_ago: int):
    """

    Args:
        mins_ago: minutes ago to get table from

    Returns:
        A joined tables of transactions with lots of useful things
    """
    _ = ig_service.create_session()
    tdelta = timedelta(seconds=secs_ago)
    now_date = datetime.now().date()

    milliseconds = int(tdelta.total_seconds() * 1000)
    transactions = ig_service.fetch_transaction_history_by_type_and_period(
        trans_type="ALL_DEAL", milliseconds=milliseconds
    )

    transactions_filtered = transactions.drop(
        # Drop this date since activity date is more acurate
        ["date", "period", "transactionType", "cashTransaction"],
        axis=1,
    )
    transactions_filtered = transactions_filtered.rename(
        columns={"reference": "closing_dealId"}
    ).astype({"openLevel": float, "closeLevel": float})

    transactions_filtered["y_true"] = (
        transactions_filtered["closeLevel"] / transactions_filtered["openLevel"]
    )
    transactions_filtered["profitAndLoss_numeric"] = (
        transactions_filtered["profitAndLoss"].str.removeprefix("£").astype(float)
    )
    transactions_filtered = transactions_filtered[
        [
            "closing_dealId",
            "instrumentName",
            "openLevel",
            "closeLevel",
            "y_true",
            "size",
            "profitAndLoss",
            "profitAndLoss_numeric",
            "currency",
        ]
    ]

    ## ACTIVITY
    activity = ig_service.fetch_account_activity(
        from_date=now_date - timedelta(days=(tdelta.days + 1)),
        to_date=now_date + timedelta(days=1),
        detailed=True,
    )
    ## WE ARE ONLY LOOKING AT SELL ACTIVITY
    sell_activity_filtered = activity[activity.direction == "SELL"][
        [
            "date",
            # "epic",
            "dealId",
            "affectedDealId",
        ]
    ]
    sell_activity_filtered = sell_activity_filtered.rename(
        columns={
            "dealId": "closing_dealId",
            "affectedDealId": "opening_dealId",
            "date": "sell_date",
        }
    )
    sell_activity_filtered.sell_date = pd.to_datetime(sell_activity_filtered.sell_date)
    sell_activity_filtered = sell_activity_filtered[
        [
            "opening_dealId",  # We get this from the affected deal
            "closing_dealId",  # They all have closing deal IDs because we filtered direction==SELL and size =1.0 for all
            "sell_date",
            # "epic",
        ]
    ]

    # Maybe we could get the buy_date from the opening_dealId in activity
    buy_activity_filtered = activity[activity.direction == "BUY"][
        [
            "date",
            "dealId",
            # "affectedDealId",
        ]
    ].rename(
        columns={
            "dealId": "opening_dealId",
            "date": "buy_date",
        }
    )

    activity_filtered = sell_activity_filtered.merge(
        buy_activity_filtered, on="opening_dealId"
    )

    position_metrics = pd.read_csv(TMP_DIR / "position_metrics.csv")
    position_metrics = position_metrics.rename(columns={"dealId": "opening_dealId"})
    position_metrics = position_metrics[
        [
            "opening_dealId",
            "y_pred",
            "model_used",
            # "buy_date", # Maybe this should be renamed to buy_date_real or something and there should be some maning consistency
        ]
    ]

    joined = transactions_filtered.merge(
        activity_filtered, left_on="closing_dealId", right_on="closing_dealId"
    ).merge(position_metrics, left_on="opening_dealId", right_on="opening_dealId")

    reorder = [
        "opening_dealId",
        "closing_dealId",
        "sell_date",
        "buy_date",
        "openLevel",
        "closeLevel",
        "instrumentName",
        "profitAndLoss",
        "profitAndLoss_numeric",
        "size",
        "currency",
        "model_used",
        "y_pred",
        "y_true",
    ]

    joined = joined[reorder]
    joined = joined.set_index("sell_date")
    joined = joined.sort_index(ascending=True)
    joined.to_csv(TMP_DIR / "transactions_joined.csv")
    return None


def parse_trade_stream(item):
    "Maybe just persist the whole stream to a csv and then use queries to get what we need"
    OPU_dict = item["values"]["OPU"]
    df = pd.DataFrame(
        dealId=OPU_dict["dealId"],
        dealIdOrigin=OPU_dict["dealIdOrigin"],
        direction=OPU_dict["direction"],
        epic=OPU_dict["epic"],
        status=OPU_dict["status"],
        dealStatus=OPU_dict["dealStatus"],
        level=OPU_dict["level"],
        size=OPU_dict["size"],
        timestamp=OPU_dict["timestamp"],
    )
    return df


if __name__ == "__main__":
    # write_to_transations_joined(10)
    parse_trade_stream()
