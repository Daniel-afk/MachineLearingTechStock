import os
import yfinance as yf
import pandas as pd

from config import TICKERS, START_DATE, END_DATE, DATA_DIR


def fetch_stock_data(ticker, start=START_DATE, end=END_DATE):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
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
