import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import RANDOM_SEED, RESULTS_DIR
from src.sports_features import SPORTS_FEATURE_COLS


def train_sports_rf(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_sports_xgb(X_train, y_train) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_sports_model(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")
    brier = brier_score_loss(y_test, proba)
    print(f"  {name}: acc={acc:.4f}  AUC={auc:.4f}  Brier={brier:.4f}")
    return {"accuracy": acc, "auc": auc, "brier": brier, "predictions": preds, "probabilities": proba}


def walk_forward_sports(feat_df: pd.DataFrame, model_type: str = "xgboost", n_splits: int = 4) -> dict:
    feat_df = feat_df.sort_values("date").reset_index(drop=True)
    X = feat_df[SPORTS_FEATURE_COLS].values
    y = feat_df["label"].values.astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    all_preds = np.full(len(X), -1, dtype=int)
    all_proba = np.full(len(X), np.nan)

    for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = train_sports_xgb(X_tr, y_tr) if model_type == "xgboost" else train_sports_rf(X_tr, y_tr)
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, preds)
        all_preds[te_idx] = preds
        all_proba[te_idx] = proba

        fold_rows.append({
            "Fold": fold_i + 1,
            "N train": len(tr_idx),
            "N test": len(te_idx),
            "Accuracy": round(acc, 4),
        })
        print(f"  Fold {fold_i+1}/{n_splits}: acc={acc:.4f}")

    mask = all_preds >= 0
    oos_acc = accuracy_score(y[mask], all_preds[mask])
    return {
        "folds": pd.DataFrame(fold_rows),
        "oos_accuracy": oos_acc,
        "predictions": all_preds,
        "probabilities": all_proba,
    }


def train_and_save_sports(league: str, feat_df: pd.DataFrame):
    """Train RF + XGBoost on all available data and save models."""
    if len(feat_df) < 100:
        print(f"  {league}: only {len(feat_df)} games, skipping")
        return

    feat_df = feat_df.sort_values("date").reset_index(drop=True)
    X = feat_df[SPORTS_FEATURE_COLS].values
    y = feat_df["label"].values.astype(int)

    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[:split])
    X_te = scaler.transform(X[split:])
    y_tr, y_te = y[:split], y[split:]

    print(f"  Training RF...")
    rf = train_sports_rf(X_tr, y_tr)
    evaluate_sports_model(rf, X_te, y_te, f"RF [{league}]")

    print(f"  Training XGBoost...")
    xgb = train_sports_xgb(X_tr, y_tr)
    evaluate_sports_model(xgb, X_te, y_te, f"XGBoost [{league}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, f"scaler_sports_{league}.joblib"))
    joblib.dump(rf,     os.path.join(RESULTS_DIR, f"rf_sports_{league}.joblib"))
    joblib.dump(xgb,    os.path.join(RESULTS_DIR, f"xgb_sports_{league}.joblib"))
    print(f"  Saved models for {league}")
