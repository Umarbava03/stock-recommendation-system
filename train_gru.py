from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU


DATA_FILE = Path("data/training_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "best_gru_classifier.keras"
SCALER_FILE = MODEL_DIR / "best_gru_classifier_scaler.joblib"
ENCODER_FILE = MODEL_DIR / "best_gru_label_encoder.joblib"

SEQUENCE_LENGTH = 30
FEATURE_COLUMNS = [
    "price_change_pct",
    "volatility",
    "avg_volume",
    "ma_5",
    "ma_10",
    "trend_signal",
    "ma_20",
    "momentum_5",
    "volatility_5",
    "price_vs_ma_10",
    "rsi_14",
]
TARGET_COLUMN = "target_binary"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)


def build_sequences(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X, y = [], []

    for symbol in df["symbol"].unique():
        symbol_df = df[df["symbol"] == symbol].copy()
        feature_values = symbol_df[feature_cols].values
        target_values = symbol_df[target_col].values

        for i in range(SEQUENCE_LENGTH, len(symbol_df)):
            X.append(feature_values[i - SEQUENCE_LENGTH:i])
            y.append(target_values[i])

    return np.array(X), np.array(y)


def train_gru() -> None:
    df = load_data()

    split_index = int(len(df) * 0.8)
    split_date = df["as_of_date"].sort_values().iloc[split_index]

    train_df = df[df["as_of_date"] <= split_date].copy()
    test_df = df[df["as_of_date"] > split_date].copy()

    scaler = MinMaxScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    label_encoder = LabelEncoder()
    train_df[TARGET_COLUMN] = label_encoder.fit_transform(train_df[TARGET_COLUMN])
    test_df[TARGET_COLUMN] = label_encoder.transform(test_df[TARGET_COLUMN])

    X_train, y_train = build_sequences(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    X_test, y_test = build_sequences(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(len(label_encoder.classes_), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0,
    )

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    labels = label_encoder.classes_
    decoded_y_test = label_encoder.inverse_transform(y_test)
    decoded_y_pred = label_encoder.inverse_transform(y_pred)

    print("GRU Classifier Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report")
    print(classification_report(decoded_y_test, decoded_y_pred))

    print("\nConfusion Matrix")
    print(
        pd.DataFrame(
            confusion_matrix(decoded_y_test, decoded_y_pred, labels=labels),
            index=labels,
            columns=labels,
        )
    )

    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Scaler saved to: {SCALER_FILE}")
    print(f"Label encoder saved to: {ENCODER_FILE}")


if __name__ == "__main__":
    train_gru()
