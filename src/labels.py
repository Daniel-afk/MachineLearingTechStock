import numpy as np
import pandas as pd

from config import FORWARD_DAYS, BUY_THRESHOLD, SELL_THRESHOLD

LABEL_NAMES = {0: "Sell", 1: "Hold", 2: "Buy"}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    forward_return = df["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    conditions = [
        forward_return > BUY_THRESHOLD,
        forward_return < SELL_THRESHOLD,
    ]
    choices = [2, 0]  # Buy=2, Sell=0, Hold=1
    df["label"] = np.select(conditions, choices, default=1)
    df["forward_return"] = forward_return
    return df
