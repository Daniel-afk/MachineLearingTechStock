import numpy as np
import pandas as pd

from config import (
    BUY_THRESHOLD,
    CRYPTO_BUY_THRESHOLD,
    CRYPTO_FORWARD_DAYS,
    CRYPTO_SELL_THRESHOLD,
    CRYPTO_TICKERS,
    FORWARD_DAYS,
    SELL_THRESHOLD,
)

LABEL_NAMES = {0: "Sell", 1: "Hold", 2: "Buy"}


def add_labels(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    df = df.copy()
    is_crypto = ticker in CRYPTO_TICKERS
    fwd   = CRYPTO_FORWARD_DAYS if is_crypto else FORWARD_DAYS
    buy_t = CRYPTO_BUY_THRESHOLD if is_crypto else BUY_THRESHOLD
    sel_t = CRYPTO_SELL_THRESHOLD if is_crypto else SELL_THRESHOLD

    forward_return = df["Close"].pct_change(fwd).shift(-fwd)
    conditions = [forward_return > buy_t, forward_return < sel_t]
    df["label"]          = np.select(conditions, [2, 0], default=1)
    df["forward_return"] = forward_return
    return df
