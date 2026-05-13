import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from config import RANDOM_SEED, SEQUENCE_LEN

tf.random.set_seed(RANDOM_SEED)


def build_lstm(input_shape, num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_sequences(X, y, seq_len=SEQUENCE_LEN):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm(input_shape)

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    return model


def evaluate_lstm(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report

    preds = model.predict(X_test, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, target_names=["Sell", "Hold", "Buy"], zero_division=0
    )
    print(f"\n{'='*50}")
    print(f"LSTM — Test Accuracy: {acc:.4f}")
    print(report)
    return preds, acc
