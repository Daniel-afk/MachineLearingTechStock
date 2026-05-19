import numpy as np
import pandas as pd

SPORTS_FEATURE_COLS = [
    "home_win_rate_l10",
    "away_win_rate_l10",
    "home_pts_scored_l10",
    "away_pts_scored_l10",
    "home_pts_allowed_l10",
    "away_pts_allowed_l10",
    "home_win_streak",
    "away_win_streak",
    "home_rest_days",
    "away_rest_days",
    "home_home_win_rate",
    "away_away_win_rate",
    "h2h_home_win_rate",
    "elo_diff",
]

_K = 32  # Elo K-factor


def _update_elo(elo: dict, home: str, away: str, home_win: int) -> dict:
    elo = dict(elo)
    r_h = elo.get(home, 1500)
    r_a = elo.get(away, 1500)
    exp_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
    elo[home] = r_h + _K * (home_win - exp_h)
    elo[away] = r_a + _K * ((1 - home_win) - (1 - exp_h))
    return elo


def build_game_features(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute per-game rolling team features. Returns feature matrix + labels."""
    if games.empty:
        return pd.DataFrame()

    games = games.sort_values("date").reset_index(drop=True)

    team_stats: dict[str, list] = {}  # team → list of game dicts
    h2h: dict[tuple, list] = {}  # (home, away) → list of home_win
    elo: dict[str, float] = {}

    rows = []

    for _, row in games.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_win = row["home_win"]

        # Pull team histories
        h_hist = team_stats.get(home, [])
        a_hist = team_stats.get(away, [])

        def recent(hist, n=window):
            return hist[-n:] if len(hist) >= 1 else []

        def win_rate(hist):
            if not hist:
                return 0.5
            return sum(g["win"] for g in hist) / len(hist)

        def pts_avg(hist, key):
            if not hist:
                return 0.0
            return np.mean([g[key] for g in hist])

        def streak(hist):
            if not hist:
                return 0
            s, last = 0, hist[-1]["win"]
            for g in reversed(hist):
                if g["win"] == last:
                    s += 1
                else:
                    break
            return s if last else -s

        def rest(hist):
            if not hist:
                return 7.0
            return min((row["date"] - hist[-1]["date"]).days, 30)

        def home_wr(hist):
            h = [g for g in hist if g.get("is_home")]
            return win_rate(h) if h else 0.5

        def away_wr(hist):
            a = [g for g in hist if not g.get("is_home")]
            return win_rate(a) if a else 0.5

        h_rec = recent(h_hist)
        a_rec = recent(a_hist)
        pair_hist = h2h.get((home, away), [])

        feat = {
            "home_win_rate_l10": win_rate(h_rec),
            "away_win_rate_l10": win_rate(a_rec),
            "home_pts_scored_l10": pts_avg(h_rec, "pts_scored"),
            "away_pts_scored_l10": pts_avg(a_rec, "pts_scored"),
            "home_pts_allowed_l10": pts_avg(h_rec, "pts_allowed"),
            "away_pts_allowed_l10": pts_avg(a_rec, "pts_allowed"),
            "home_win_streak": streak(h_hist),
            "away_win_streak": streak(a_hist),
            "home_rest_days": rest(h_hist),
            "away_rest_days": rest(a_hist),
            "home_home_win_rate": home_wr(h_hist),
            "away_away_win_rate": away_wr(a_hist),
            "h2h_home_win_rate": (sum(pair_hist) / len(pair_hist)) if pair_hist else 0.5,
            "elo_diff": elo.get(home, 1500) - elo.get(away, 1500),
            "label": home_win,
            "date": row["date"],
            "home_team": home,
            "away_team": away,
        }
        rows.append(feat)

        # Update state AFTER computing features (no lookahead)
        for team, is_home, won, scored, allowed in [
            (home, True, home_win, row["home_score"], row["away_score"]),
            (away, False, 1 - home_win, row["away_score"], row["home_score"]),
        ]:
            if team not in team_stats:
                team_stats[team] = []
            team_stats[team].append({
                "win": won,
                "pts_scored": scored,
                "pts_allowed": allowed,
                "is_home": is_home,
                "date": row["date"],
            })

        h2h.setdefault((home, away), []).append(home_win)
        elo = _update_elo(elo, home, away, home_win)

    return pd.DataFrame(rows)


def features_for_matchup(
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date=None,
    window: int = 10,
) -> np.ndarray | None:
    """Return a feature vector for a single upcoming matchup."""
    if history.empty:
        return None
    if game_date is None:
        game_date = history["date"].max() + pd.Timedelta(days=7)

    past = history[history["date"] < game_date].copy()

    def recent_team(team, is_home_flag):
        played = past[(past["home_team"] == team) | (past["away_team"] == team)].tail(window)
        return played, is_home_flag

    h_games = past[(past["home_team"] == home_team) | (past["away_team"] == home_team)].tail(window)
    a_games = past[(past["home_team"] == away_team) | (past["away_team"] == away_team)].tail(window)

    def wr(df, team):
        if df.empty:
            return 0.5
        wins = ((df["home_team"] == team) & (df["label"] == 1)) | ((df["away_team"] == team) & (df["label"] == 0))
        return wins.mean()

    def avg_pts(df, team, scored=True):
        if df.empty:
            return 0.0
        as_home = df["home_team"] == team
        vals = np.where(as_home, df["home_score"] if scored else df["away_score"],
                        df["away_score"] if scored else df["home_score"])
        return float(np.mean(vals))

    def streak(df, team):
        if df.empty:
            return 0
        wins = [1 if ((r["home_team"] == team and r["label"] == 1) or
                      (r["away_team"] == team and r["label"] == 0)) else 0
                for _, r in df.iterrows()]
        if not wins:
            return 0
        s, last = 0, wins[-1]
        for w in reversed(wins):
            if w == last:
                s += 1
            else:
                break
        return s if last else -s

    def rest(df, team):
        if df.empty:
            return 7.0
        last = df["date"].max()
        return min((game_date - last).days, 30) if pd.notnull(last) else 7.0

    def venue_wr(df, team, as_home):
        sub = df[df["home_team"] == team] if as_home else df[df["away_team"] == team]
        if sub.empty:
            return 0.5
        wins = (sub["label"] == 1).mean() if as_home else (sub["label"] == 0).mean()
        return float(wins)

    h2h = past[(past["home_team"] == home_team) & (past["away_team"] == away_team)]
    h2h_wr = h2h["label"].mean() if not h2h.empty else 0.5

    # Elo: rebuild from full history up to this game
    elo: dict[str, float] = {}
    for _, r in past.iterrows():
        elo = _update_elo(elo, r["home_team"], r["away_team"], r["label"])
    elo_diff = elo.get(home_team, 1500) - elo.get(away_team, 1500)

    vec = [
        wr(h_games, home_team),
        wr(a_games, away_team),
        avg_pts(h_games, home_team, scored=True),
        avg_pts(a_games, away_team, scored=True),
        avg_pts(h_games, home_team, scored=False),
        avg_pts(a_games, away_team, scored=False),
        streak(h_games, home_team),
        streak(a_games, away_team),
        rest(h_games, home_team),
        rest(a_games, away_team),
        venue_wr(past, home_team, as_home=True),
        venue_wr(past, away_team, as_home=False),
        h2h_wr,
        elo_diff,
    ]
    return np.array(vec, dtype=float)
