import os
import sys
import warnings

# When run directly with `python app.py`, re-launch via streamlit so the
# browser opens automatically — no need to know about `streamlit run`.
if __name__ == "__main__" and "streamlit" not in sys.modules:
    import subprocess
    sys.exit(subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__,
         "--server.headless", "false"],
        check=False,
    ).returncode)

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CRYPTO_TICKERS, RESULTS_DIR, SEQUENCE_LEN, TICKERS
from src.backtest import run_backtest
from src.data_fetcher import fetch_stock_data
from src.features import FEATURE_COLS, add_features
from src.fundamentals import fetch_fundamentals
from src.labels import add_labels
from src.model_lstm import TENSORFLOW_AVAILABLE
from src.news_fetcher import daily_sentiment, fetch_news
from src.sports_features import SPORTS_FEATURE_COLS, build_game_features, features_for_matchup
from src.sports_fetcher import SPORTS_CONFIG, fetch_historical_games, fetch_upcoming_games
from src.live_data import fetch_live_quote, fetch_intraday, fetch_live_scores

LABEL_NAME  = {0: "Sell", 1: "Hold", 2: "Buy"}
LABEL_COLOR = {0: "#ef4444", 1: "#94a3b8", 2: "#22c55e"}
LABEL_MARKER = {0: "triangle-down", 1: "circle", 2: "triangle-up"}
SENTIMENT_COLOR = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}
SENTIMENT_ICON  = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}

st.set_page_config(
    page_title="ML Stock & Crypto Signals",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 ML Signals Dashboard")
    asset_class = st.radio("Asset class", ["Stocks", "Crypto", "Sports"], horizontal=True)

    if asset_class == "Sports":
        sport_league = st.selectbox("League", list(SPORTS_CONFIG.keys()))
        ticker = None
        ticker_list = []
        model_name = "XGBoost"
    else:
        ticker_list = TICKERS if asset_class == "Stocks" else CRYPTO_TICKERS
        ticker = st.selectbox("Ticker", ticker_list)
        available_models = ["Random Forest", "XGBoost"] + (["LSTM"] if TENSORFLOW_AVAILABLE else [])
        model_name = st.selectbox("Model", available_models)
        if not TENSORFLOW_AVAILABLE:
            st.caption("LSTM unavailable — TensorFlow not supported on Python 3.14+.")
        sport_league = None

    if asset_class != "Sports":
        lookback_days = st.slider("Chart history (days)", 60, 500, 252)
        signal_days   = st.slider("Signal history (days)", 10, 90, 30)
    if asset_class == "Crypto":
        st.info("Crypto uses ±5% thresholds over 3-day windows (vs ±2% / 5-day for stocks).")

    st.divider()
    live_refresh = st.toggle("🔴 Live mode", value=False,
                             help="Auto-refreshes live prices and scores every 30 seconds")
    if live_refresh:
        refresh_secs = st.select_slider("Refresh interval", [15, 30, 60, 120], value=30)
    else:
        refresh_secs = None

    st.divider()
    if asset_class == "Sports":
        train_btn = st.button("🏆 Train Sports Models", use_container_width=True)
        st.caption(
            "Fetches historical game data from ESPN (NFL, NBA, MLB, NHL) "
            "and trains a separate model per sport. Runs `python sports_main.py`."
        )
    else:
        train_btn = st.button("🚀 Train / Retrain Models", use_container_width=True)
        st.caption(
            "Trains a **separate model for each asset** — AAPL learns only from AAPL, "
            "BTC only from BTC, etc. Covers all 13 assets in one run (~10–15 min)."
        )

# ── Training ─────────────────────────────────────────────────────────────────

if train_btn:
    script = "sports_main.py" if asset_class == "Sports" else "main.py"
    with st.spinner(f"Training… check your terminal for progress. (`{script}`)"):
        import subprocess
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True,
        )
    if result.returncode == 0:
        st.success("Training complete! Models saved to `results/`.")
        if result.stdout:
            with st.expander("📋 Training log", expanded=False):
                st.code(result.stdout[-4000:])
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    else:
        st.error("Training failed.")
        if result.stdout:
            with st.expander("📋 Training log"):
                st.code(result.stdout[-4000:])
        st.code(result.stderr[-3000:])

# ── Auto-refresh (live mode) ──────────────────────────────────────────────────

if live_refresh and refresh_secs:
    import time as _time
    _time.sleep(refresh_secs)
    st.rerun()

# ── Cached loaders ────────────────────────────────────────────────────────────

def _safe(t: str) -> str:
    return t.replace("-", "_")

# Historical price data barely changes — cache for 6 hours.
@st.cache_data(ttl=21600, show_spinner="Fetching data…")
def load_ticker(t):
    try:
        df = fetch_stock_data(t)
        df = add_features(df)
        fund = fetch_fundamentals(t, df)
        df = df.join(fund, how="left")
        df = add_labels(df, t)
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to load {t}: {type(exc).__name__}: {exc}") from exc

# News expires faster — cache for 30 minutes.
@st.cache_data(ttl=1800, show_spinner=False)
def load_news(t):
    return fetch_news(t, max_items=30)

@st.cache_resource(show_spinner=False)
def load_scaler(t):
    path = os.path.join(RESULTS_DIR, f"scaler_{_safe(t)}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_rf(t):
    path = os.path.join(RESULTS_DIR, f"rf_{_safe(t)}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_xgb(t):
    path = os.path.join(RESULTS_DIR, f"xgb_{_safe(t)}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

def get_model(name, t):
    if name == "Random Forest": return load_rf(t)
    if name == "XGBoost":       return load_xgb(t)
    return None  # LSTM not trained per-ticker

# ── Sports-specific loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_sports_scaler(league: str):
    path = os.path.join(RESULTS_DIR, f"scaler_sports_{league}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_sports_rf(league: str):
    path = os.path.join(RESULTS_DIR, f"rf_sports_{league}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_sports_xgb(league: str):
    path = os.path.join(RESULTS_DIR, f"xgb_sports_{league}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data(ttl=3600, show_spinner="Fetching historical game data…")
def load_sports_history(league: str) -> pd.DataFrame:
    games = fetch_historical_games(league)
    if games.empty:
        return pd.DataFrame()
    return build_game_features(games).dropna().reset_index(drop=True)

@st.cache_data(ttl=1800, show_spinner="Fetching upcoming games…")
def load_upcoming_games(league: str) -> pd.DataFrame:
    return fetch_upcoming_games(league)

# ── Sports dashboard ──────────────────────────────────────────────────────────

if asset_class == "Sports":
    st.markdown(f"## 🏆 {sport_league} — Game Predictions")

    rf_ready  = os.path.exists(os.path.join(RESULTS_DIR, f"rf_sports_{sport_league}.joblib"))
    xgb_ready = os.path.exists(os.path.join(RESULTS_DIR, f"xgb_sports_{sport_league}.joblib"))
    models_ready = rf_ready and xgb_ready

    if not models_ready:
        st.warning(
            f"No trained models found for **{sport_league}**. "
            "Click **Train Sports Models** in the sidebar to fetch game data and train the models. "
            "This will download historical game results from ESPN (free, no API key)."
        )

    # Show feature explanation
    with st.expander("ℹ️ How sports predictions work", expanded=False):
        st.markdown("""
### What the model predicts
The model predicts whether the **home team will win** each game (binary: Win / Loss).
Win probabilities are shown so you can see how confident the model is.

### Features used
| Feature | Description |
|---|---|
| Win rate (last 10) | Home / away team recent win % |
| Points scored/allowed | Avg pts over last 10 games |
| Win streak | Current consecutive wins (positive) or losses (negative) |
| Rest days | Days since last game (captures fatigue / rust) |
| Home/away record | Venue-specific win rate |
| Head-to-head | Historical win rate in this exact matchup |
| Elo rating diff | Dynamic skill rating updated after each game |

### Models
- **Random Forest** and **XGBoost** are trained per sport (one model per league)
- Training uses a **time-ordered 80/20 split** — the test set is always future games
- Walk-forward validation with 4 folds gives an honest out-of-sample accuracy estimate

### Data source
Historical game results come from the **ESPN public API** (no key required).
Seasons 2018–2024 are used for training.

⚠️ **Disclaimer:** Sports outcomes are inherently uncertain. This is for research and
entertainment only. Never use these predictions for gambling or financial decisions.
        """)

    # Metric row
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("League", sport_league)
    col_b.metric("Models trained", "✅ Yes" if models_ready else "❌ No")

    history_df = load_sports_history(sport_league)
    col_c.metric("Historical games", f"{len(history_df):,}" if not history_df.empty else "—")

    # ── Live scores ───────────────────────────────────────────────────────────
    st.subheader("📡 Today's scoreboard")
    with st.spinner("Fetching live scores…"):
        live_scores = fetch_live_scores(sport_league)

    if live_scores.empty:
        st.info("No games on the ESPN scoreboard right now.")
    else:
        live_games   = live_scores[live_scores["status_raw"] == "in"]
        sched_games  = live_scores[live_scores["status_raw"] == "pre"]
        final_games  = live_scores[live_scores["status_raw"] == "post"]

        # Live games — prominent cards
        if not live_games.empty:
            st.markdown("#### 🔴 In Progress")
            cols = st.columns(min(len(live_games), 4))
            for i, (_, g) in enumerate(live_games.iterrows()):
                with cols[i % len(cols)]:
                    st.markdown(
                        f"**{g['away_team']}** {g['away_score']} — "
                        f"{g['home_score']} **{g['home_team']}**  \n"
                        f"<span style='color:#ef4444; font-size:0.82em'>{g['period']}  {g['clock']}</span>",
                        unsafe_allow_html=True,
                    )

        # Scheduled
        if not sched_games.empty:
            st.markdown("#### 📅 Scheduled")
            sched_display = sched_games[["date", "away_team", "home_team"]].rename(
                columns={"date": "Date", "away_team": "Away", "home_team": "Home"}
            )
            st.dataframe(sched_display, use_container_width=True, hide_index=True)

        # Final scores
        if not final_games.empty:
            st.markdown("#### ✅ Final")
            final_display = final_games[["away_team", "away_score", "home_score", "home_team", "winner"]].rename(
                columns={"away_team": "Away", "away_score": "Away score",
                         "home_score": "Home score", "home_team": "Home", "winner": "Winner"}
            )
            st.dataframe(final_display, use_container_width=True, hide_index=True)

        if live_refresh:
            st.caption(f"🔴 Live mode · refreshing every {refresh_secs}s")
        else:
            if st.button("⟳ Refresh scores"):
                st.rerun()

    st.divider()

    # Upcoming games predictions
    st.subheader("Upcoming games")
    upcoming = load_upcoming_games(sport_league)

    if upcoming.empty:
        st.info("No upcoming games found. The ESPN scoreboard may be between seasons.")
    elif not models_ready or history_df.empty:
        st.dataframe(upcoming[["date", "home_team", "away_team"]].rename(
            columns={"date": "Date", "home_team": "Home", "away_team": "Away"}
        ), use_container_width=True)
        st.info("Train models to see win probability predictions.")
    else:
        xgb_model = load_sports_xgb(sport_league)
        sports_scaler = load_sports_scaler(sport_league)

        pred_rows = []
        for _, game in upcoming.iterrows():
            feat_vec = features_for_matchup(
                history_df, game["home_team"], game["away_team"], game["date"]
            )
            if feat_vec is None or sports_scaler is None or xgb_model is None:
                home_prob = 0.5
            else:
                scaled = sports_scaler.transform(feat_vec.reshape(1, -1))
                home_prob = float(xgb_model.predict_proba(scaled)[0, 1])

            away_prob = 1.0 - home_prob
            pred = "🏠 Home" if home_prob >= 0.5 else "✈️ Away"
            conf = max(home_prob, away_prob)
            conf_label = "High" if conf >= 0.65 else ("Medium" if conf >= 0.55 else "Low")

            pred_rows.append({
                "Date": str(game["date"])[:10],
                "Home team": game["home_team"],
                "Away team": game["away_team"],
                "Home win %": f"{home_prob*100:.0f}%",
                "Away win %": f"{away_prob*100:.0f}%",
                "Prediction": pred,
                "Confidence": conf_label,
            })

        pred_df = pd.DataFrame(pred_rows)
        st.dataframe(pred_df, use_container_width=True, height=420)

        # Win probability bar chart
        if len(pred_rows) > 0:
            st.subheader("Win probability — upcoming games")
            labels = [f"{r['Home team']} vs {r['Away team']}" for r in pred_rows]
            home_probs = [float(r["Home win %"].strip("%")) for r in pred_rows]
            away_probs = [float(r["Away win %"].strip("%")) for r in pred_rows]

            fig_sp = go.Figure()
            fig_sp.add_trace(go.Bar(
                name="Home win %", x=labels, y=home_probs,
                marker_color="#22c55e", text=[f"{p:.0f}%" for p in home_probs],
                textposition="inside",
            ))
            fig_sp.add_trace(go.Bar(
                name="Away win %", x=labels, y=away_probs,
                marker_color="#6366f1", text=[f"{p:.0f}%" for p in away_probs],
                textposition="inside",
            ))
            fig_sp.update_layout(
                barmode="stack", template="plotly_dark", height=380,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis_title="Win probability (%)",
                legend=dict(orientation="h", y=1.05),
            )
            fig_sp.add_hline(y=50, line=dict(color="white", dash="dash", width=0.8))
            st.plotly_chart(fig_sp, use_container_width=True)

    # Recent results table
    st.divider()
    st.subheader("Recent results")
    if not history_df.empty:
        recent_games = history_df.sort_values("date", ascending=False).head(20)
        display_cols = ["date", "home_team", "away_team", "label"]
        recent_display = recent_games[display_cols].copy()
        recent_display["Result"] = recent_display["label"].map({1: "🏠 Home win", 0: "✈️ Away win"})
        recent_display = recent_display.rename(columns={
            "date": "Date", "home_team": "Home", "away_team": "Away"
        })[["Date", "Home", "Away", "Result"]]
        st.dataframe(recent_display, use_container_width=True, height=360)
    else:
        st.info("No game history loaded yet.")

    st.stop()

# ── Load price data (needed for chart) — news loads later ────────────────────

df     = load_ticker(ticker)
scaler = load_scaler(ticker)
model  = get_model(model_name, ticker)

# ── Predictions ───────────────────────────────────────────────────────────────

def predict_signals(df, model, scaler, model_name):
    valid = df.dropna(subset=FEATURE_COLS)
    if model is None or scaler is None or len(valid) == 0:
        return pd.Series(dtype=int)
    X = scaler.transform(valid[FEATURE_COLS].values)
    if model_name == "LSTM":
        if len(X) <= SEQUENCE_LEN:
            return pd.Series(dtype=int)
        seqs = np.array([X[i - SEQUENCE_LEN:i] for i in range(SEQUENCE_LEN, len(X))])
        preds = model.predict(seqs, verbose=0).argmax(axis=1)
        return pd.Series(preds, index=valid.index[SEQUENCE_LEN:])
    preds = model.predict(X)
    return pd.Series(preds, index=valid.index)

signals        = predict_signals(df, model, scaler, model_name)
plot_df        = df.iloc[-lookback_days:].copy()
recent_signals = signals[signals.index.isin(plot_df.index)]

# ── Live quote banner ─────────────────────────────────────────────────────────

with st.container():
    live_quote = fetch_live_quote(ticker)
    if live_quote:
        price      = live_quote["price"]
        chg        = live_quote["change"]
        chg_pct    = live_quote["change_pct"]
        fetched_at = live_quote["fetched_at"].strftime("%H:%M:%S UTC")
        color      = "#22c55e" if chg >= 0 else "#ef4444"
        arrow      = "▲" if chg >= 0 else "▼"
        mktcap     = live_quote.get("market_cap")
        mktcap_str = (
            f"${mktcap/1e12:.2f}T" if mktcap and mktcap >= 1e12 else
            f"${mktcap/1e9:.1f}B"  if mktcap and mktcap >= 1e9  else ""
        )
        st.markdown(
            f"<div style='background:#1e293b; border-radius:8px; padding:10px 18px; "
            f"display:flex; align-items:center; gap:28px; margin-bottom:12px'>"
            f"<span style='font-size:1.5em; font-weight:700'>{ticker}</span>"
            f"<span style='font-size:1.8em; font-weight:700'>${price:,.4f}</span>"
            f"<span style='font-size:1.2em; color:{color}'>"
            f"{arrow} {abs(chg):+.4f} ({chg_pct:+.2f}%)</span>"
            + (f"<span style='color:#94a3b8; font-size:0.9em'>Mkt cap {mktcap_str}</span>" if mktcap_str else "")
            + f"<span style='color:#64748b; font-size:0.8em; margin-left:auto'>"
            f"{'🔴 Live · ' if live_refresh else ''}Updated {fetched_at}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if not live_refresh and st.button("⟳ Refresh quote", key="refresh_quote"):
            st.rerun()

# ── Intraday chart (today's price action) ─────────────────────────────────────

intraday_df = fetch_intraday(ticker)
if not intraday_df.empty and "Close" in intraday_df.columns:
    with st.expander("📡 Today's intraday price (5-min bars)", expanded=False):
        fig_id = go.Figure()
        fig_id.add_trace(go.Scatter(
            x=intraday_df.index, y=intraday_df["Close"],
            mode="lines", name="Price",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        ))
        fig_id.update_layout(
            template="plotly_dark", height=220,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time", yaxis_title="Price",
            showlegend=False,
        )
        st.plotly_chart(fig_id, use_container_width=True)

# ── How it works ──────────────────────────────────────────────────────────────

with st.expander("ℹ️ How the ML works", expanded=False):
    is_crypto = ticker in CRYPTO_TICKERS
    fwd_days  = 3 if is_crypto else 5
    threshold = "5%" if is_crypto else "2%"
    st.markdown(f"""
### What this app does
This dashboard uses **machine learning** to predict whether a stock or crypto asset
is likely to go **Up (Buy)**, **Down (Sell)**, or **stay flat (Hold)** over the next
**{fwd_days} trading days**.  It is trained on historical price data from 2018 to 2024
and evaluates each day using technical indicators and fundamental data.

---

### Step 1 — Data
Every day of price history (Open, High, Low, Close, Volume) is downloaded from
**Yahoo Finance** using `yfinance`. For stocks, quarterly earnings data (P/E ratio,
revenue growth, profit margin, debt/equity, ROE, P/B ratio) is also fetched and
forward-filled so the model knows the financial health of the company.
Crypto assets skip the fundamental step since they have no earnings.

### Step 2 — Features (what the model sees)
The raw price data is transformed into **26 signals** the model can learn from:

| Category | Features |
|---|---|
| Trend | SMA 20, SMA 50, EMA 12, EMA 26, price-to-SMA ratios |
| Momentum | MACD, MACD signal, MACD histogram, RSI(14) |
| Volatility | Bollinger Bands (upper/lower/% position), ATR(14) |
| Volume | On-Balance Volume (OBV), volume ratio vs 20-day avg |
| Returns | 1-day, 5-day, 10-day, 20-day price returns |
| Fundamentals | P/E, P/B, revenue growth, profit margin, debt/equity, ROE *(stocks only)* |

### Step 3 — Labels (what the model is trying to predict)
Each day is labelled based on what the price actually did **{fwd_days} days later**:

| Label | Condition |
|---|---|
| 🟢 **Buy** | Price rose more than **+{threshold}** |
| 🔴 **Sell** | Price fell more than **−{threshold}** |
| 🟡 **Hold** | Price moved less than **±{threshold}** |

{"*Crypto uses wider ±5% thresholds over 3 days because it is far more volatile than stocks.*" if is_crypto else ""}

### Step 4 — Models
**Each asset gets its own dedicated model** — AAPL's model only learned from AAPL
data, BTC's only from BTC, and so on. This means each model is specialised to
that asset's unique behaviour and volatility regime.

Two model types are trained per asset:

- **Random Forest** — builds hundreds of decision trees and votes. Fast, robust,
  resistant to overfitting. Good at capturing non-linear relationships in tabular data.
- **XGBoost** — a boosted tree ensemble that learns from its own mistakes
  iteration by iteration. Usually the most accurate on structured financial data.

### Step 5 — Walk-forward validation
Instead of a simple 80/20 train/test split, the model is also evaluated with
**walk-forward validation** — the data is divided into 5 time-ordered folds,
and the model is always tested on future data it has never seen during training.
This gives a much more honest estimate of real-world performance.

### Step 6 — Backtest
The predicted signals are turned into a simulated trading strategy:
- **Buy signal** → invest full portfolio at today's close
- **Sell signal** → sell entire position at today's close
- **Hold** → do nothing
- A **0.1% transaction cost** is applied to every trade

The equity curve and drawdown chart show how $10,000 would have grown (or
shrunk) compared to simply buying and holding the asset for the same period.

---

⚠️ **Disclaimer:** This is a research and education tool. Past model performance
does not guarantee future returns. Do not use these signals as financial advice.
    """)

# ── Metric cards (news sentiment filled in later via placeholder) ─────────────

st.markdown(f"## {ticker}  —  {model_name} signals")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    latest_close = plot_df["Close"].iloc[-1]
    prev_close   = plot_df["Close"].iloc[-2]
    delta_pct    = (latest_close / prev_close - 1) * 100
    st.metric("Latest Close", f"${latest_close:.2f}", f"{delta_pct:+.2f}%")
with col2:
    sig_label = LABEL_NAME[int(recent_signals.iloc[-1])] if len(recent_signals) else "—"
    st.metric("Latest Signal", sig_label)
with col3:
    st.metric("RSI (14)", f"{plot_df['RSI_14'].iloc[-1]:.1f}")
with col4:
    sentiment_placeholder = st.empty()   # filled after news loads
with col5:
    rf_ready  = os.path.exists(os.path.join(RESULTS_DIR, f"rf_{_safe(ticker)}.joblib"))
    xgb_ready = os.path.exists(os.path.join(RESULTS_DIR, f"xgb_{_safe(ticker)}.joblib"))
    ready = sum([rf_ready, xgb_ready])
    st.metric("Models trained", f"{ready} / 2", f"for {ticker}")

if model is None:
    st.warning(
        f"**{model_name}** model not found. Click **Train / Retrain Models** in the sidebar "
        "or run `python main.py` in your terminal first."
    )

# ── Main chart (price only — no news dependency) ──────────────────────────────

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2],
    vertical_spacing=0.04,
    subplot_titles=("Price & Signals", "RSI (14)", "MACD"),
)

fig.add_trace(go.Candlestick(
    x=plot_df.index,
    open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"],   close=plot_df["Close"],
    name="OHLC",
    increasing_line_color="#22c55e",
    decreasing_line_color="#ef4444",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["BB_upper"],
    line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot"),
    name="BB Upper", showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["BB_lower"],
    line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot"),
    fill="tonexty", fillcolor="rgba(148,163,184,0.06)",
    name="BB Lower", showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["SMA_20"],
    line=dict(color="#f59e0b", width=1), name="SMA 20",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["SMA_50"],
    line=dict(color="#6366f1", width=1), name="SMA 50",
), row=1, col=1)

if len(recent_signals):
    for label_id, sym in LABEL_MARKER.items():
        mask = recent_signals == label_id
        if not mask.any():
            continue
        idx = recent_signals[mask].index
        y_pos = plot_df.loc[idx, "Low"] * 0.98 if label_id == 0 else plot_df.loc[idx, "High"] * 1.02
        fig.add_trace(go.Scatter(
            x=idx, y=y_pos, mode="markers",
            marker=dict(symbol=sym, size=10, color=LABEL_COLOR[label_id],
                        line=dict(width=1, color="white")),
            name=LABEL_NAME[label_id],
        ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["RSI_14"],
    line=dict(color="#a78bfa", width=1.5), name="RSI",
), row=2, col=1)
fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=0.8), row=2, col=1)
fig.add_hline(y=30, line=dict(color="#22c55e", dash="dash", width=0.8), row=2, col=1)
fig.add_hrect(y0=30, y1=70, fillcolor="rgba(148,163,184,0.05)", line_width=0, row=2, col=1)

hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in plot_df["MACD_hist"]]
fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df["MACD_hist"],
    name="MACD Hist", marker_color=hist_colors, showlegend=False,
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["MACD"],
    line=dict(color="#38bdf8", width=1.2), name="MACD",
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["MACD_signal"],
    line=dict(color="#fb923c", width=1.2), name="Signal",
), row=3, col=1)

fig.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="RSI",       row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="MACD",      row=3, col=1)

# Chart renders immediately — no news needed
st.plotly_chart(fig, use_container_width=True)

# ── Signals table  |  Feature importance ──────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"Last {signal_days} days — signals")
    if len(recent_signals) == 0:
        st.info("No signals yet. Train models first.")
    else:
        recent = recent_signals.iloc[-signal_days:].copy()
        tbl = df.loc[recent.index, ["Close", "RSI_14", "forward_return"]].copy()
        tbl["Signal"]     = recent.map(LABEL_NAME)
        tbl["Fwd Return"] = tbl["forward_return"].map(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
        tbl["RSI"]        = tbl["RSI_14"].map(lambda x: f"{x:.1f}")
        tbl["Close"]      = tbl["Close"].map(lambda x: f"${x:.2f}")
        tbl = tbl[["Close", "RSI", "Signal", "Fwd Return"]].iloc[::-1]

        def highlight_signal(row):
            c = {"Buy": "color: #22c55e", "Sell": "color: #ef4444", "Hold": "color: #94a3b8"}
            return ["", "", c.get(row["Signal"], ""), ""]

        st.dataframe(tbl.style.apply(highlight_signal, axis=1),
                     use_container_width=True, height=420)

with col_right:
    st.subheader("Feature importance")
    m = load_rf(ticker) if model_name == "Random Forest" else load_xgb(ticker)
    display_name = model_name
    if m is None:
        st.info("Train models to see feature importance.")
    else:
        feat_df = (
            pd.DataFrame({"Feature": FEATURE_COLS, "Importance": m.feature_importances_})
            .sort_values("Importance", ascending=True).tail(15)
        )
        fig_imp = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"],
            orientation="h", marker_color="#6366f1",
        ))
        fig_imp.update_layout(
            template="plotly_dark", height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Importance", title=f"{display_name} — top features",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── Backtest ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Backtest — Strategy vs Buy & Hold")

from config import TEST_SIZE
if model is None or scaler is None:
    st.info("Train models first to see backtest results.")
elif len(signals) < 20:
    st.info("Not enough signal data. Train models first.")
else:
    n_test = int(len(df) * TEST_SIZE)
    test_df_bt = df.iloc[-n_test:]
    test_sigs   = signals[signals.index.isin(test_df_bt.index)]
    test_prices = test_df_bt["Close"].reindex(test_sigs.index)

    bt = run_backtest(test_sigs, test_prices)

    # ── Metric row ────────────────────────────────────────────────────────────
    alpha = bt.total_return - bt.benchmark_return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategy return",  f"{bt.total_return*100:+.1f}%",
              f"α {alpha*100:+.1f}% vs B&H")
    c2.metric("Buy & Hold",       f"{bt.benchmark_return*100:+.1f}%")
    c3.metric("Sharpe ratio",     f"{bt.sharpe_ratio:.2f}")
    c4.metric("Max drawdown",     f"{bt.max_drawdown*100:.1f}%")
    c5.metric("Win rate",         f"{bt.win_rate*100:.0f}%",
              f"{bt.num_trades} trades")

    # ── Equity curve + Drawdown subplots ─────────────────────────────────────
    drawdown = (bt.equity_curve - bt.equity_curve.cummax()) / bt.equity_curve.cummax() * 100

    fig_bt = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio value ($)", "Drawdown (%)"),
    )

    # Equity curves
    fig_bt.add_trace(go.Scatter(
        x=bt.equity_curve.index, y=bt.equity_curve.round(2),
        name="Strategy",
        line=dict(color="#22c55e", width=2.5),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.06)",
    ), row=1, col=1)
    fig_bt.add_trace(go.Scatter(
        x=bt.benchmark_curve.index, y=bt.benchmark_curve.round(2),
        name="Buy & Hold",
        line=dict(color="#6366f1", width=2, dash="dash"),
    ), row=1, col=1)

    # Trade markers on equity curve
    if not bt.trades.empty:
        buys  = bt.trades[bt.trades["action"] == "BUY"]
        sells = bt.trades[bt.trades["action"] == "SELL"]
        if not buys.empty:
            fig_bt.add_trace(go.Scatter(
                x=buys["date"],
                y=bt.equity_curve.reindex(buys["date"]).values,
                mode="markers", name="Buy entry",
                marker=dict(symbol="triangle-up", size=11, color="#22c55e",
                            line=dict(width=1.5, color="white")),
            ), row=1, col=1)
        if not sells.empty:
            fig_bt.add_trace(go.Scatter(
                x=sells["date"],
                y=bt.equity_curve.reindex(sells["date"]).values,
                mode="markers", name="Sell exit",
                marker=dict(symbol="triangle-down", size=11, color="#ef4444",
                            line=dict(width=1.5, color="white")),
            ), row=1, col=1)

    # Drawdown area
    fig_bt.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.round(2),
        name="Drawdown",
        line=dict(color="#ef4444", width=1),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        showlegend=False,
    ), row=2, col=1)
    fig_bt.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1), row=2, col=1)

    fig_bt.update_layout(
        height=550,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis2_rangeslider_visible=False,
    )
    fig_bt.update_yaxes(title_text="Value ($)",    row=1, col=1)
    fig_bt.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    st.plotly_chart(fig_bt, use_container_width=True)
    st.caption(
        f"Test period: {test_df_bt.index[0].date()} → {test_df_bt.index[-1].date()}  ·  "
        f"$10,000 initial capital  ·  0.1% transaction cost per trade  ·  no shorting"
    )

# ── Live news feed — loads after chart is visible ─────────────────────────────

st.divider()
st.subheader(f"📰 Latest news — {ticker}")

with st.spinner("Loading news…"):
    news_df    = load_news(ticker)
    sent_daily = daily_sentiment(news_df)

# Fill in the sentiment metric card now that news is loaded
if not news_df.empty:
    avg_score = news_df["sentiment_score"].mean()
    label = "Positive" if avg_score >= 0.05 else ("Negative" if avg_score <= -0.05 else "Neutral")
    sentiment_placeholder.metric(
        "News Sentiment",
        f"{SENTIMENT_ICON[label]} {label}",
        f"{avg_score:+.2f}",
    )
else:
    sentiment_placeholder.metric("News Sentiment", "—")

if news_df.empty:
    st.info("No news articles found.")
else:
    # Compact sentiment trend bar chart
    if not sent_daily.empty:
        recent_sent = sent_daily.tail(30)
        bar_colors  = ["#22c55e" if v >= 0.05 else ("#ef4444" if v <= -0.05 else "#94a3b8")
                       for v in recent_sent.values]
        fig_sent = go.Figure(go.Bar(
            x=[pd.Timestamp(d) for d in recent_sent.index],
            y=recent_sent.values,
            marker_color=bar_colors,
        ))
        fig_sent.update_layout(
            template="plotly_dark", height=180,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Sentiment score",
            showlegend=False,
        )
        fig_sent.add_hline(y=0, line=dict(color="white", width=0.5))
        st.plotly_chart(fig_sent, use_container_width=True)

    for _, row in news_df.iterrows():
        icon      = SENTIMENT_ICON[row["sentiment"]]
        color     = SENTIMENT_COLOR[row["sentiment"]]
        ts        = row["datetime"]
        time_str  = ts.strftime("%b %d, %H:%M UTC") if pd.notna(ts) else ""
        title     = row["title"] or "—"
        link      = row["link"] or ""
        publisher = row["publisher"] or ""
        headline  = f"[{title}]({link})" if link else title
        st.markdown(
            f"{icon} &nbsp; **{headline}**  \n"
            f"<span style='color:{color}; font-size:0.8em'>{row['sentiment']} "
            f"({row['sentiment_score']:+.2f})</span>"
            f"<span style='color:#64748b; font-size:0.8em'>&nbsp;·&nbsp;{publisher}"
            f"&nbsp;·&nbsp;{time_str}</span>",
            unsafe_allow_html=True,
        )

# ── Model comparison image ────────────────────────────────────────────────────

cmp_path = os.path.join(RESULTS_DIR, "model_comparison.png")
if os.path.exists(cmp_path):
    st.divider()
    st.subheader("Model comparison — confusion matrices")
    st.image(cmp_path)
