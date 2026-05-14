import os

import pandas as pd
import yfinance as yf

from config import DATA_DIR, END_DATE, START_DATE, TICKERS

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance column formats across all versions.

    yfinance <0.2.38  → flat string columns  e.g. 'Close'
    yfinance >=0.2.38 → MultiIndex (Price, Ticker) e.g. ('Close', 'AAPL')
    yfinance 1.x      → MultiIndex (Price, Ticker) same shape
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Take the first level which holds the field name (Close, High, …)
        df.columns = df.columns.get_level_values(0)

    # Normalise casing – yfinance sometimes returns lower-case in 1.x
    rename = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=rename)
    return df


def fetch_stock_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    df = _flatten_columns(raw)

    # Keep only standard OHLCV columns that are present
    cols = [c for c in _OHLCV if c in df.columns]
    df = df[cols].copy()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path)
    return df


def load_all_tickers(tickers=TICKERS):
    frames = []
    for ticker in tickers:
        df = fetch_stock_data(ticker)
        df["Ticker"] = ticker
        frames.append(df)
    return pd.concat(frames)
