import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CRYPTO_TICKERS, DATA_DIR, RANDOM_SEED, RESULTS_DIR, TEST_SIZE, TICKERS
from src.backtest import run_backtest
from src.data_fetcher import fetch_stock_data
from src.features import FEATURE_COLS, add_features
from src.fundamentals import fetch_fundamentals
from src.labels import add_labels
from src.model_rf import evaluate_model, train_random_forest, train_xgboost
from src.walk_forward import run_walk_forward


def safe_name(ticker: str) -> str:
    return ticker.replace("-", "_")


def _prepare_ticker(ticker: str) -> pd.DataFrame:
    df = fetch_stock_data(ticker)
    df = add_features(df)
    fund = fetch_fundamentals(ticker, df)
    df = df.join(fund, how="left")
    df = add_labels(df, ticker)
    return df.dropna(subset=FEATURE_COLS + ["label"])


def _train_ticker(ticker: str, data: pd.DataFrame) -> list:
    from sklearn.preprocessing import StandardScaler

    if len(data) < 200:
        print(f"  Skipping — only {len(data)} rows")
        return []

    split_idx = int(len(data) * (1 - TEST_SIZE))
    train_df, test_df = data.iloc[:split_idx], data.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values.astype(int)
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values.astype(int)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, f"scaler_{safe_name(ticker)}.joblib"))

    results = []

    print(f"  Training Random Forest...")
    rf = train_random_forest(X_tr_sc, y_train)
    rf_preds, rf_acc = evaluate_model(rf, X_te_sc, y_test, f"Random Forest [{ticker}]")
    joblib.dump(rf, os.path.join(RESULTS_DIR, f"rf_{safe_name(ticker)}.joblib"))
    results.append(("Random Forest", y_test, rf_preds, rf_acc))

    print(f"  Training XGBoost...")
    xgb = train_xgboost(X_tr_sc, y_train)
    xgb_preds, xgb_acc = evaluate_model(xgb, X_te_sc, y_test, f"XGBoost [{ticker}]")
    joblib.dump(xgb, os.path.join(RESULTS_DIR, f"xgb_{safe_name(ticker)}.joblib"))
    results.append(("XGBoost", y_test, xgb_preds, xgb_acc))

    return results


def _walk_and_backtest(ticker: str, data: pd.DataFrame):
    print(f"  Walk-forward validation (4 folds)...")
    wf = run_walk_forward(data, model_type="xgboost", n_splits=4)
    print(f"  OOS accuracy: {wf['oos_accuracy']:.4f}")
    print(wf["folds"].to_string(index=False))

    valid = data.reset_index(drop=False)
    date_col = valid.columns[0]
    preds = wf["predictions"]
    oos = preds >= 0
    dates  = valid.loc[oos.values, date_col].values
    sigs   = pd.Series(preds[oos].values,              index=dates)
    prices = pd.Series(valid.loc[oos.values, "Close"].values, index=dates)

    bt = run_backtest(sigs, prices)
    alpha = bt.total_return - bt.benchmark_return
    print(
        f"  Backtest → strategy {bt.total_return*100:+.1f}%  "
        f"B&H {bt.benchmark_return*100:+.1f}%  "
        f"α {alpha*100:+.1f}%  "
        f"Sharpe {bt.sharpe_ratio:+.2f}  "
        f"MaxDD {bt.max_drawdown*100:.1f}%  "
        f"trades {bt.num_trades}  win {bt.win_rate*100:.0f}%"
    )


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    summary = []

    for ticker in TICKERS + CRYPTO_TICKERS:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        data = _prepare_ticker(ticker)
        dist = data["label"].value_counts().sort_index().rename({0: "Sell", 1: "Hold", 2: "Buy"})
        print(f"  Rows: {len(data)}  |  Labels: {dict(dist)}")

        results = _train_ticker(ticker, data)
        for name, _, _, acc in results:
            summary.append({"Ticker": ticker, "Model": name, "Test Accuracy": round(acc, 4)})

        if results:
            _walk_and_backtest(ticker, data)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
