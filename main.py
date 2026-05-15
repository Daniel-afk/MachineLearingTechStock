import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, SEQUENCE_LEN, TEST_SIZE, TICKERS, CRYPTO_TICKERS
from src.backtest import run_backtest
from src.data_fetcher import fetch_stock_data
from src.evaluate import plot_comparison, print_summary
from src.features import FEATURE_COLS, add_features
from src.fundamentals import fetch_fundamentals
from src.labels import add_labels
from src.model_lstm import create_sequences, evaluate_lstm, train_lstm
from src.model_rf import evaluate_model, train_random_forest, train_xgboost
from src.walk_forward import run_walk_forward


def _prepare_data():
    frames = []
    all_tickers = TICKERS + CRYPTO_TICKERS
    print("Fetching and processing stock + crypto data...")
    for ticker in all_tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        df = fetch_stock_data(ticker)
        df = add_features(df)
        fund = fetch_fundamentals(ticker, df)
        df = df.join(fund, how="left")
        df = add_labels(df, ticker)
        df["Ticker"] = ticker
        frames.append(df)
        print("done")

    data = pd.concat(frames).sort_index()
    data = data.dropna(subset=FEATURE_COLS + ["label"])
    return data


def _time_split(data: pd.DataFrame):
    data = data.sort_index()
    split_idx = int(len(data) * (1 - TEST_SIZE))
    return data.iloc[:split_idx], data.iloc[split_idx:]


def _build_lstm_sequences(train_df, test_df, scaler):
    train_X_seqs, train_y_seqs = [], []
    test_X_seqs, test_y_seqs = [], []

    for ticker in TICKERS:
        t_train = train_df[train_df["Ticker"] == ticker]
        t_test = test_df[test_df["Ticker"] == ticker]

        if len(t_train) <= SEQUENCE_LEN or len(t_test) == 0:
            continue

        Xt = scaler.transform(t_train[FEATURE_COLS].values)
        yt = t_train["label"].values.astype(int)
        Xe = scaler.transform(t_test[FEATURE_COLS].values)
        ye = t_test["label"].values.astype(int)

        Xs_tr, ys_tr = create_sequences(Xt, yt, SEQUENCE_LEN)
        train_X_seqs.append(Xs_tr)
        train_y_seqs.append(ys_tr)

        combined_X = np.vstack([Xt, Xe])
        combined_y = np.concatenate([yt, ye])
        Xs_all, ys_all = create_sequences(combined_X, combined_y, SEQUENCE_LEN)
        Xs_te = Xs_all[len(ys_tr):]
        ys_te = ys_all[len(ys_tr):]

        test_X_seqs.append(Xs_te)
        test_y_seqs.append(ys_te)

    return (
        np.concatenate(train_X_seqs),
        np.concatenate(train_y_seqs),
        np.concatenate(test_X_seqs),
        np.concatenate(test_y_seqs),
    )


def _print_backtest(ticker, signals, prices):
    if len(signals) < 10:
        return
    bt = run_backtest(signals, prices)
    alpha = bt.total_return - bt.benchmark_return
    print(
        f"  {ticker:<5}  strategy {bt.total_return*100:+6.1f}%  "
        f"B&H {bt.benchmark_return*100:+6.1f}%  "
        f"alpha {alpha*100:+5.1f}%  "
        f"Sharpe {bt.sharpe_ratio:+.2f}  "
        f"MaxDD {bt.max_drawdown*100:.1f}%  "
        f"trades {bt.num_trades}  win {bt.win_rate*100:.0f}%"
    )


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    data = _prepare_data()
    print(f"\nTotal samples: {len(data)}")
    dist = data["label"].value_counts().sort_index().rename({0: "Sell", 1: "Hold", 2: "Buy"})
    print(f"Label distribution:\n{dist}\n")

    train_df, test_df = _time_split(data)
    print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")

    from sklearn.preprocessing import StandardScaler

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values.astype(int)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values.astype(int)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.joblib"))

    results = []

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train_sc, y_train)
    rf_preds, rf_acc = evaluate_model(rf, X_test_sc, y_test, "Random Forest")
    joblib.dump(rf, os.path.join(RESULTS_DIR, "rf_model.joblib"))
    results.append(("Random Forest", y_test, rf_preds, rf_acc))

    print("\nTraining XGBoost...")
    xgb = train_xgboost(X_train_sc, y_train)
    xgb_preds, xgb_acc = evaluate_model(xgb, X_test_sc, y_test, "XGBoost")
    joblib.dump(xgb, os.path.join(RESULTS_DIR, "xgb_model.joblib"))
    results.append(("XGBoost", y_test, xgb_preds, xgb_acc))

    from src.model_lstm import TENSORFLOW_AVAILABLE
    if TENSORFLOW_AVAILABLE:
        print("\nBuilding LSTM sequences...")
        Xtr, ytr, Xte, yte = _build_lstm_sequences(train_df, test_df, scaler)
        print(f"LSTM sequences — train: {Xtr.shape}, test: {Xte.shape}")
        val_cut = int(0.1 * len(Xtr))
        Xval, yval = Xtr[-val_cut:], ytr[-val_cut:]
        Xtr2, ytr2 = Xtr[:-val_cut], ytr[:-val_cut]
        print("\nTraining LSTM...")
        lstm = train_lstm(Xtr2, ytr2, Xval, yval, epochs=30, batch_size=64)
        lstm.save(os.path.join(RESULTS_DIR, "lstm_model.keras"))
        lstm_preds, lstm_acc = evaluate_lstm(lstm, Xte, yte)
        results.append(("LSTM", yte, lstm_preds, lstm_acc))
    else:
        print("\nSkipping LSTM — TensorFlow not installed.")

    print_summary(results)
    plot_comparison(results)

    # ── Walk-forward validation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION  (XGBoost, 5 folds)")
    print("=" * 60)
    wf = run_walk_forward(data, model_type="xgboost", n_splits=5)
    print(f"\nOverall OOS accuracy: {wf['oos_accuracy']:.4f}")
    print("\nFold breakdown:")
    print(wf["folds"].to_string(index=False))

    # ── Backtest on walk-forward OOS signals ──────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST  (walk-forward OOS signals, $10,000 initial capital)")
    print("=" * 60)
    for ticker in TICKERS + CRYPTO_TICKERS:
        ticker_mask = data["Ticker"] == ticker
        ticker_data = data[ticker_mask]
        oos_sigs = wf["predictions"].reindex(ticker_data.index)
        valid = oos_sigs[oos_sigs >= 0]
        _print_backtest(ticker, valid, ticker_data.loc[valid.index, "Close"])


if __name__ == "__main__":
    main()
