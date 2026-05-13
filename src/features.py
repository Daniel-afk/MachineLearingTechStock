import numpy as np
import pandas as pd


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["RSI_14"] = _rsi(close, 14)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    band_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_pct"] = (close - df["BB_lower"]) / band_range

    df["ATR_14"] = _atr(high, low, close, 14)

    df["OBV"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    df["Vol_ratio"] = volume / vol_ma

    for n in [1, 5, 10, 20]:
        df[f"Return_{n}d"] = close.pct_change(n)

    df["Price_SMA20_ratio"] = close / df["SMA_20"].replace(0, np.nan)
    df["Price_SMA50_ratio"] = close / df["SMA_50"].replace(0, np.nan)

    return df


FEATURE_COLS = [
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "MACD", "MACD_signal", "MACD_hist",
    "RSI_14",
    "BB_upper", "BB_lower", "BB_pct",
    "ATR_14", "OBV", "Vol_ratio",
    "Return_1d", "Return_5d", "Return_10d", "Return_20d",
    "Price_SMA20_ratio", "Price_SMA50_ratio",
]
