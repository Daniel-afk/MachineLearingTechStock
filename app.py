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

from config import FEATURE_COLS, RESULTS_DIR, SEQUENCE_LEN, TICKERS
from src.data_fetcher import fetch_stock_data
from src.features import add_features
from src.labels import add_labels
from src.model_lstm import TENSORFLOW_AVAILABLE
from src.news_fetcher import daily_sentiment, fetch_news

LABEL_NAME = {0: "Sell", 1: "Hold", 2: "Buy"}
LABEL_COLOR = {0: "#ef4444", 1: "#94a3b8", 2: "#22c55e"}
LABEL_MARKER = {0: "triangle-down", 1: "circle", 2: "triangle-up"}
SENTIMENT_COLOR = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}
SENTIMENT_ICON = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}

st.set_page_config(
    page_title="ML Tech Stock Signals",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Tech Stock ML")
    ticker = st.selectbox("Stock", TICKERS)
    available_models = ["Random Forest", "XGBoost"] + (["LSTM"] if TENSORFLOW_AVAILABLE else [])
    model_name = st.selectbox("Model", available_models)
    if not TENSORFLOW_AVAILABLE:
        st.caption("LSTM unavailable — TensorFlow not supported on Python 3.14+.")
    lookback_days = st.slider("Chart history (days)", 60, 500, 252)
    signal_days = st.slider("Signal history (days)", 10, 90, 30)
    show_sentiment = st.toggle("Show sentiment overlay", value=True)
    st.divider()
    train_btn = st.button("🚀 Train / Retrain Models", use_container_width=True)
    st.caption("Training downloads data and fits all 3 models. Takes ~5–10 min.")

# ── Training ─────────────────────────────────────────────────────────────────

if train_btn:
    with st.spinner("Training Random Forest, XGBoost, and LSTM… check your terminal for progress."):
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True,
        )
    if result.returncode == 0:
        st.success("Training complete! Models saved to `results/`.")
        st.cache_resource.clear()
        st.cache_data.clear()
    else:
        st.error("Training failed.")
        st.code(result.stderr[-3000:])

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching stock data…")
def load_ticker(t):
    df = fetch_stock_data(t)
    df = add_features(df)
    df = add_labels(df)
    return df


@st.cache_data(ttl=900, show_spinner="Fetching latest news…")
def load_news(t):
    return fetch_news(t, max_items=30)


@st.cache_resource(show_spinner=False)
def load_scaler():
    path = os.path.join(RESULTS_DIR, "scaler.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_rf():
    path = os.path.join(RESULTS_DIR, "rf_model.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_xgb():
    path = os.path.join(RESULTS_DIR, "xgb_model.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_lstm():
    path = os.path.join(RESULTS_DIR, "lstm_model.keras")
    if not os.path.exists(path):
        return None
    try:
        from tensorflow import keras
        return keras.models.load_model(path)
    except Exception:
        return None


def get_model(name):
    if name == "Random Forest":
        return load_rf()
    if name == "XGBoost":
        return load_xgb()
    return load_lstm()


df = load_ticker(ticker)
news_df = load_news(ticker)
scaler = load_scaler()
model = get_model(model_name)
sent_daily = daily_sentiment(news_df)

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


signals = predict_signals(df, model, scaler, model_name)
plot_df = df.iloc[-lookback_days:].copy()
recent_signals = signals[signals.index.isin(plot_df.index)]

# ── Metric cards ──────────────────────────────────────────────────────────────

st.markdown(f"## {ticker}  —  {model_name} signals")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    latest_close = plot_df["Close"].iloc[-1]
    prev_close = plot_df["Close"].iloc[-2]
    delta_pct = (latest_close / prev_close - 1) * 100
    st.metric("Latest Close", f"${latest_close:.2f}", f"{delta_pct:+.2f}%")

with col2:
    if len(recent_signals):
        latest_sig = int(recent_signals.iloc[-1])
        st.metric("Latest Signal", LABEL_NAME[latest_sig])
    else:
        st.metric("Latest Signal", "—")

with col3:
    rsi_val = plot_df["RSI_14"].iloc[-1]
    st.metric("RSI (14)", f"{rsi_val:.1f}")

with col4:
    if not news_df.empty:
        avg_score = news_df["sentiment_score"].mean()
        label = "Positive" if avg_score >= 0.05 else ("Negative" if avg_score <= -0.05 else "Neutral")
        st.metric("News Sentiment", f"{SENTIMENT_ICON[label]} {label}", f"{avg_score:+.2f}")
    else:
        st.metric("News Sentiment", "—")

with col5:
    total = 3 if TENSORFLOW_AVAILABLE else 2
    models_ready = sum([
        os.path.exists(os.path.join(RESULTS_DIR, "rf_model.joblib")),
        os.path.exists(os.path.join(RESULTS_DIR, "xgb_model.joblib")),
        os.path.exists(os.path.join(RESULTS_DIR, "lstm_model.keras")),
    ])
    st.metric("Models trained", f"{models_ready} / {total}")

if model is None:
    st.warning(
        f"**{model_name}** model not found. Click **Train / Retrain Models** in the sidebar "
        "or run `python main.py` in your terminal first."
    )

# ── Main chart ────────────────────────────────────────────────────────────────

n_rows = 4 if show_sentiment and not sent_daily.empty else 3
row_heights = [0.5, 0.17, 0.17, 0.16] if n_rows == 4 else [0.6, 0.2, 0.2]
subplot_titles = (
    ("Price & Signals", "News Sentiment", "RSI (14)", "MACD")
    if n_rows == 4 else
    ("Price & Signals", "RSI (14)", "MACD")
)

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    vertical_spacing=0.04,
    subplot_titles=subplot_titles,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=plot_df.index,
    open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"], close=plot_df["Close"],
    name="OHLC", increasing_line_color="#22c55e",
    decreasing_line_color="#ef4444",
), row=1, col=1)

# Bollinger Bands
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

# Signal markers
if len(recent_signals):
    for label_id, sym in LABEL_MARKER.items():
        mask = recent_signals == label_id
        if not mask.any():
            continue
        idx = recent_signals[mask].index
        price_offset = (
            plot_df.loc[idx, "Low"] * 0.98
            if label_id == 0
            else plot_df.loc[idx, "High"] * 1.02
        )
        fig.add_trace(go.Scatter(
            x=idx, y=price_offset, mode="markers",
            marker=dict(symbol=sym, size=10, color=LABEL_COLOR[label_id],
                        line=dict(width=1, color="white")),
            name=LABEL_NAME[label_id],
        ), row=1, col=1)

# ── Sentiment overlay row ─────────────────────────────────────────────────────

sentiment_row = 2
rsi_row = 3 if n_rows == 4 else 2
macd_row = 4 if n_rows == 4 else 3

if n_rows == 4:
    sent_plot = sent_daily[
        sent_daily.index >= plot_df.index.date.min()
    ]
    bar_colors = [
        "#22c55e" if v >= 0.05 else ("#ef4444" if v <= -0.05 else "#94a3b8")
        for v in sent_plot.values
    ]
    fig.add_trace(go.Bar(
        x=[pd.Timestamp(d) for d in sent_plot.index],
        y=sent_plot.values,
        name="Daily Sentiment",
        marker_color=bar_colors,
        showlegend=False,
    ), row=sentiment_row, col=1)
    fig.add_hline(y=0, line=dict(color="white", width=0.5), row=sentiment_row, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["RSI_14"],
    line=dict(color="#a78bfa", width=1.5), name="RSI",
), row=rsi_row, col=1)
fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=0.8), row=rsi_row, col=1)
fig.add_hline(y=30, line=dict(color="#22c55e", dash="dash", width=0.8), row=rsi_row, col=1)
fig.add_hrect(y0=30, y1=70, fillcolor="rgba(148,163,184,0.05)", line_width=0, row=rsi_row, col=1)

# MACD
hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in plot_df["MACD_hist"]]
fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df["MACD_hist"],
    name="MACD Hist", marker_color=hist_colors, showlegend=False,
), row=macd_row, col=1)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["MACD"],
    line=dict(color="#38bdf8", width=1.2), name="MACD",
), row=macd_row, col=1)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["MACD_signal"],
    line=dict(color="#fb923c", width=1.2), name="Signal",
), row=macd_row, col=1)

fig.update_layout(
    height=750,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
if n_rows == 4:
    fig.update_yaxes(title_text="Sentiment", row=sentiment_row, col=1)
fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, range=[0, 100])
fig.update_yaxes(title_text="MACD", row=macd_row, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Signals table  |  Feature importance ──────────────────────────────────────

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"Last {signal_days} days — signals")
    if len(recent_signals) == 0:
        st.info("No signals yet. Train models first.")
    else:
        recent = recent_signals.iloc[-signal_days:].copy()
        tbl = df.loc[recent.index, ["Close", "RSI_14", "forward_return"]].copy()
        tbl["Signal"] = recent.map(LABEL_NAME)
        tbl["Fwd Return"] = tbl["forward_return"].map(
            lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—"
        )
        tbl["RSI"] = tbl["RSI_14"].map(lambda x: f"{x:.1f}")
        tbl["Close"] = tbl["Close"].map(lambda x: f"${x:.2f}")
        tbl = tbl[["Close", "RSI", "Signal", "Fwd Return"]].iloc[::-1]

        def highlight_signal(row):
            c = {"Buy": "color: #22c55e", "Sell": "color: #ef4444", "Hold": "color: #94a3b8"}
            return ["", "", c.get(row["Signal"], ""), ""]

        st.dataframe(
            tbl.style.apply(highlight_signal, axis=1),
            use_container_width=True,
            height=420,
        )

with col_right:
    st.subheader("Feature importance")
    m = load_rf() if model_name in ("Random Forest", "LSTM") else load_xgb()
    display_name = "Random Forest" if model_name in ("Random Forest", "LSTM") else "XGBoost"
    if m is None:
        st.info("Train models to see feature importance.")
    else:
        feat_df = (
            pd.DataFrame({"Feature": FEATURE_COLS, "Importance": m.feature_importances_})
            .sort_values("Importance", ascending=True)
            .tail(15)
        )
        fig_imp = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"],
            orientation="h", marker_color="#6366f1",
        ))
        fig_imp.update_layout(
            template="plotly_dark", height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Importance", yaxis_title="",
            title=f"{display_name} — top features",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── Live news feed ────────────────────────────────────────────────────────────

st.divider()
st.subheader(f"📰 Latest news — {ticker}")

if news_df.empty:
    st.info("No news articles found.")
else:
    for _, row in news_df.iterrows():
        icon = SENTIMENT_ICON[row["sentiment"]]
        color = SENTIMENT_COLOR[row["sentiment"]]
        ts = row["datetime"]
        time_str = ts.strftime("%b %d, %H:%M UTC") if pd.notna(ts) else ""
        title = row["title"] or "—"
        link = row["link"] or ""
        publisher = row["publisher"] or ""

        headline = f"[{title}]({link})" if link else title
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
