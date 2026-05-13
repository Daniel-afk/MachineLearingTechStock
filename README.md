# ML Tech Stock Signal Classifier

Predict **Buy / Hold / Sell** signals for major tech stocks using Random Forest, XGBoost, and LSTM models trained on technical indicators.

## Stocks covered
AAPL, MSFT, GOOGL, NVDA, META, AMZN, TSLA

## How it works

1. **Data** — Downloads OHLCV history (2018–2024) via `yfinance` and caches to `data/`
2. **Features** — Computes 20 technical indicators per day: SMA/EMA, MACD, RSI, Bollinger Bands, ATR, OBV, volume ratio, price returns
3. **Labels** — 5-day forward return: Buy (>+2%), Sell (<-2%), Hold (otherwise)
4. **Models** — Trains Random Forest, XGBoost, and a 2-layer LSTM; evaluates each on a held-out time split
5. **Results** — Prints per-class precision/recall/F1 and saves a confusion-matrix comparison to `results/model_comparison.png`

## Quick start

```bash
pip install -r requirements.txt
python app.py        # opens the dashboard in your browser automatically
```

The dashboard has a **Train / Retrain Models** button — no need to run `main.py` separately.

## Browser dashboard features

- **Candlestick chart** with Bollinger Bands, SMA 20/50, and Buy/Sell/Hold signal markers
- **RSI panel** with overbought/oversold zones
- **MACD panel** with histogram
- **Signal table** — last N days with predicted label and actual forward return
- **Feature importance** bar chart (Random Forest or XGBoost)
- **Confusion matrix** comparison once training is complete
- Switch between tickers and models instantly from the sidebar

## Project layout

```
config.py          — tickers, dates, thresholds, hyperparameters
app.py             — Streamlit web dashboard
main.py            — CLI training pipeline
src/
  data_fetcher.py  — yfinance download + CSV cache
  features.py      — technical indicator computation
  labels.py        — buy/sell/hold label generation
  model_rf.py      — Random Forest & XGBoost training + evaluation
  model_lstm.py    — LSTM model, sequence builder, training + evaluation
  evaluate.py      — confusion matrix plots and summary table
data/              — cached CSV files (git-ignored)
results/           — saved models and plots
```

## Configuration

Edit `config.py` to adjust:
- `TICKERS` — which stocks to include
- `FORWARD_DAYS` — prediction horizon (default 5 trading days)
- `BUY_THRESHOLD` / `SELL_THRESHOLD` — signal thresholds (default ±2%)
- `SEQUENCE_LEN` — LSTM lookback window (default 60 days)
