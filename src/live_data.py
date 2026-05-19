"""Live / real-time data helpers — no API keys required."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"


# ── Live stock / crypto price ─────────────────────────────────────────────────

def fetch_live_quote(ticker: str) -> dict:
    """Return current price info using yfinance fast_info."""
    try:
        info = yf.Ticker(ticker).fast_info
        price      = float(info.last_price or 0)
        prev_close = float(info.previous_close or price)
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        return {
            "price":      price,
            "change":     change,
            "change_pct": change_pct,
            "currency":   getattr(info, "currency", "USD"),
            "market_cap": getattr(info, "market_cap", None),
            "volume":     getattr(info, "three_month_average_volume", None),
            "fetched_at": datetime.now(timezone.utc),
        }
    except Exception:
        return {}


def fetch_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame:
    """Return today's intraday OHLCV bars."""
    try:
        df = yf.download(
            ticker, period="1d", interval=interval,
            progress=False, auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.strip().title() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


# ── Live sports scores ────────────────────────────────────────────────────────

_SPORT_MAP = {
    "NFL": ("football", "nfl"),
    "NBA": ("basketball", "nba"),
    "MLB": ("baseball", "mlb"),
    "NHL": ("hockey", "nhl"),
}

_STATUS_LABELS = {
    "pre":   "Scheduled",
    "in":    "🔴 LIVE",
    "post":  "Final",
}


def _parse_live_event(event: dict) -> dict | None:
    comps = event.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]
    status_type = comp.get("status", {}).get("type", {})
    status_name = status_type.get("name", "pre")   # pre / in / post

    competitors = {c["homeAway"]: c for c in comp.get("competitors", [])}
    if "home" not in competitors or "away" not in competitors:
        return None

    home = competitors["home"]
    away = competitors["away"]

    # Clock / period detail (football/basketball/hockey)
    status_obj = comp.get("status", {})
    clock  = status_obj.get("displayClock", "")
    period = status_obj.get("period", 0)

    # Sport-specific period labels
    sport_period = _period_label(event, period, status_name)

    return {
        "game_id":    event["id"],
        "date":       event.get("date", "")[:10],
        "home_team":  home["team"]["abbreviation"],
        "away_team":  away["team"]["abbreviation"],
        "home_score": home.get("score", "—"),
        "away_score": away.get("score", "—"),
        "status":     _STATUS_LABELS.get(status_name, status_name),
        "status_raw": status_name,
        "clock":      clock,
        "period":     sport_period,
        "winner":     (
            home["team"]["abbreviation"] if status_name == "post" and
            int(home.get("score", 0) or 0) > int(away.get("score", 0) or 0)
            else (
                away["team"]["abbreviation"] if status_name == "post" else ""
            )
        ),
    }


def _period_label(event: dict, period: int, status_name: str) -> str:
    if status_name == "pre":
        return ""
    if status_name == "post":
        return "Final"
    if period == 0:
        return ""
    # Try to detect sport from event league
    league = event.get("league", {}).get("abbreviation", "").upper()
    if league in ("NFL",):
        q = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}.get(period, f"OT{period-4}")
        return q
    if league in ("NBA",):
        q = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}.get(period, f"OT{period-4}")
        return q
    if league in ("NHL",):
        p = {1: "1st", 2: "2nd", 3: "3rd"}.get(period, f"OT{period-3}")
        return p
    if league in ("MLB",):
        return f"Inning {period}"
    return f"Period {period}"


def fetch_live_scores(league: str) -> pd.DataFrame:
    """Fetch today's scoreboard (scheduled + live + final) for a league."""
    if league not in _SPORT_MAP:
        return pd.DataFrame()
    sport, slug = _SPORT_MAP[league]
    url = f"{_ESPN_BASE}/{sport}/{slug}/scoreboard"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    for event in data.get("events", []):
        # Inject league abbreviation so _period_label works
        event.setdefault("league", {})["abbreviation"] = league
        parsed = _parse_live_event(event)
        if parsed:
            rows.append(parsed)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort: live first, then scheduled, then final
    order = {"🔴 LIVE": 0, "Scheduled": 1, "Final": 2}
    df["_sort"] = df["status"].map(order).fillna(3)
    df = df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return df
