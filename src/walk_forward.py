import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED
from src.features import FEATURE_COLS
from src.model_rf import train_random_forest, train_xgboost


def run_walk_forward(
    data: pd.DataFrame,
    model_type: str = "xgboost",
    n_splits: int = 5,
) -> dict:
    """Walk-forward validation over the combined multi-ticker dataset.

    Returns a dict with:
      folds         — DataFrame of per-fold metrics
      oos_accuracy  — overall out-of-sample accuracy
      predictions   — Series of OOS predicted labels (index = date)
      actuals       — Series of true labels (index = date)
    """
    data = data.sort_index()
    valid = data.dropna(subset=FEATURE_COLS + ["label"])

    X = valid[FEATURE_COLS].values
    y = valid["label"].values.astype(int)
    dates = valid.index

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    all_preds = np.full(len(X), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        if model_type == "xgboost":
            model = train_xgboost(X_tr_sc, y_tr)
        else:
            model = train_random_forest(X_tr_sc, y_tr)

        preds = model.predict(X_te_sc)
        acc = accuracy_score(y_te, preds)
        all_preds[test_idx] = preds

        fold_rows.append(
            {
                "Fold": fold_i + 1,
                "Train start": dates[train_idx[0]].date(),
                "Train end": dates[train_idx[-1]].date(),
                "Test start": dates[test_idx[0]].date(),
                "Test end": dates[test_idx[-1]].date(),
                "N train": len(train_idx),
                "N test": len(test_idx),
                "Accuracy": round(acc, 4),
            }
        )
        print(
            f"  Fold {fold_i + 1}/{n_splits}: "
            f"test {dates[test_idx[0]].date()} → {dates[test_idx[-1]].date()}  "
            f"acc={acc:.4f}"
        )

    oos_mask = all_preds >= 0
    oos_acc = accuracy_score(y[oos_mask], all_preds[oos_mask])

    # Use a RangeIndex so callers can safely align via boolean masks on the
    # original valid-data slice without hitting duplicate-date errors.
    return {
        "folds": pd.DataFrame(fold_rows),
        "oos_accuracy": oos_acc,
        "predictions": pd.Series(all_preds, index=range(len(all_preds))),
        "actuals": pd.Series(y, index=range(len(y))),
    }
