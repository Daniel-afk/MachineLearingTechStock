import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

DATA_DIR = "data"

SPORTS_CONFIG = {
    "NFL": {"sport": "football", "league": "nfl", "seasons": range(2018, 2025)},
    "NBA": {"sport": "basketball", "league": "nba", "seasons": range(2018, 2025)},
    "MLB": {"sport": "baseball", "league": "mlb", "seasons": range(2018, 2025)},
    "NHL": {"sport": "hockey", "league": "nhl", "seasons": range(2018, 2025)},
}

_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _espn_get(url: str, params: dict = None) -> dict:
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _parse_event(event: dict) -> dict | None:
    comps = event.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]
    if not comp.get("status", {}).get("type", {}).get("completed", False):
        return None

    competitors = {c["homeAway"]: c for c in comp.get("competitors", [])}
    if "home" not in competitors or "away" not in competitors:
        return None

    home = competitors["home"]
    away = competitors["away"]

    try:
        home_score = int(home.get("score", 0))
        away_score = int(away.get("score", 0))
    except (ValueError, TypeError):
        return None

    return {
        "game_id": event["id"],
        "date": event["date"][:10],
        "home_team": home["team"]["abbreviation"],
        "away_team": away["team"]["abbreviation"],
        "home_score": home_score,
        "away_score": away_score,
        "home_win": int(home_score > away_score),
        "neutral_site": comp.get("neutralSite", False),
    }


def _fetch_season_games(sport: str, league: str, season: int) -> list[dict]:
    url = f"{_BASE}/{sport}/{league}/scoreboard"
    # ESPN uses season year; seasontype=2 is regular season, 3 is playoffs
    rows = []
    for season_type in (2, 3):
        page = 1
        while True:
            try:
                data = _espn_get(url, {"season": season, "seasontype": season_type, "limit": 200, "page": page})
            except Exception:
                break
            events = data.get("events", [])
            if not events:
                break
            for ev in events:
                parsed = _parse_event(ev)
                if parsed:
                    rows.append(parsed)
            # ESPN pagination
            total_pages = data.get("pageCount", 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.2)
    return rows


def fetch_historical_games(league: str) -> pd.DataFrame:
    cfg = SPORTS_CONFIG[league]
    cache_path = os.path.join(DATA_DIR, f"sports_{league}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    all_rows = []
    for season in cfg["seasons"]:
        rows = _fetch_season_games(cfg["sport"], cfg["league"], season)
        all_rows.extend(rows)
        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).drop_duplicates("game_id")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_upcoming_games(league: str) -> pd.DataFrame:
    cfg = SPORTS_CONFIG[league]
    url = f"{_BASE}/{cfg['sport']}/{cfg['league']}/scoreboard"
    rows = []
    try:
        data = _espn_get(url, {"limit": 50})
        for ev in data.get("events", []):
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            competitors = {c["homeAway"]: c for c in comp.get("competitors", [])}
            if "home" not in competitors or "away" not in competitors:
                continue
            completed = comp.get("status", {}).get("type", {}).get("completed", False)
            rows.append({
                "game_id": ev["id"],
                "date": ev["date"][:10],
                "home_team": competitors["home"]["team"]["abbreviation"],
                "away_team": competitors["away"]["team"]["abbreviation"],
                "completed": completed,
            })
    except Exception:
        pass

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["game_id", "date", "home_team", "away_team", "completed"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[~df["completed"]].reset_index(drop=True) if not df.empty else df
