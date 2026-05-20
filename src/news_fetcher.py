from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def _score(text: str) -> float:
    return _analyzer.polarity_scores(text)["compound"]


def _sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def fetch_news(ticker: str, max_items: int = 30) -> pd.DataFrame:
    raw = yf.Ticker(ticker).news or []
    rows = []
    for item in raw[:max_items]:
        content = item.get("content", {})
        title = content.get("title") or item.get("title", "")
        link = (
            content.get("canonicalUrl", {}).get("url")
            or content.get("clickThroughUrl", {}).get("url")
            or item.get("link", "")
        )
        publisher = content.get("provider", {}).get("displayName") or item.get("publisher", "")
        pub_ts = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(pub_ts, str):
            try:
                pub_dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
            except ValueError:
                pub_dt = None
        elif isinstance(pub_ts, (int, float)):
            pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
        else:
            pub_dt = None

        score = _score(title)
        rows.append({
            "datetime": pub_dt,
            "title": title,
            "publisher": publisher,
            "link": link,
            "sentiment_score": score,
            "sentiment": _sentiment_label(score),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime", ascending=False).reset_index(drop=True)


def daily_sentiment(news_df: pd.DataFrame) -> pd.Series:
    if news_df.empty or "datetime" not in news_df.columns:
        return pd.Series(dtype=float)
    tmp = news_df.dropna(subset=["datetime"]).copy()
    tmp["date"] = tmp["datetime"].dt.date
    return tmp.groupby("date")["sentiment_score"].mean().rename("sentiment")
