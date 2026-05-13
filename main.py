import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, SEQUENCE_LEN, TEST_SIZE, TICKERS
from src.data_fetcher import fetch_stock_data
from src.evaluate import plot_comparison, print_summary
from src.features import FEATURE_COLS, add_features
from src.labels import add_labels
from src.model_lstm import create_sequences, evaluate_lstm, train_lstm
from src.model_rf import evaluate_model, train_random_forest, train_xgboost


def _prepare_data():
    from sklearn.preprocessing import StandardScaler

    frames = []
    print("Fetching and processing stock data...")
    for ticker in TICKERS:
        print(f"  {ticker}...", end=" ", flush=True)
        df = fetch_stock_data(ticker)
        df = add_features(df)
        df = add_labels(df)
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

    print_summary(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
