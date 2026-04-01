from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


DATA_FILE = Path("data/training_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "best_lstm_classifier.keras"
SCALER_FILE = MODEL_DIR / "best_lstm_classifier_scaler.joblib"
ENCODER_FILE = MODEL_DIR / "best_lstm_label_encoder.joblib"

SEQUENCE_LENGTH = 20
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
    df = df.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)
    return df


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
    



def train_lstm() -> None:
    df = load_data()

    split_index = int(len(df) * 0.8)
    split_date = df["as_of_date"].sort_values().iloc[split_index]

    train_df = df[df["as_of_date"] <= split_date].copy()
    test_df = df[df["as_of_date"] > split_date].copy()

    scaler = MinMaxScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    label_encoder=LabelEncoder()
    train_df[TARGET_COLUMN]=label_encoder.fit_transform(train_df[TARGET_COLUMN])
    test_df[TARGET_COLUMN] =label_encoder.transform(test_df[TARGET_COLUMN])

    X_train, y_train = build_sequences(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    X_test, y_test = build_sequences(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(len(classes), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0,
    )

    predictions = model.predict(X_test, verbose=0)
    y_pred= np.argmax(predictions, axis= 1)

    accuracy=accuracy_score(y_test,y_pred)
    macro_f1= f1_score(y_test, y_pred, average="macro")

    labels = label_encoder.classes_
    decoded_y_pred=label_encoder.inverse_transform(y_pred)
    decoded_y_test=label_encoder.inverse_transform(y_test)

    print("LSTM Classifier Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report")
    print(classification_report(decoded_y_test, decoded_y_pred))

    print("\nConfusion Matrix")
    print(pd.DataFrame(
        confusion_matrix(decoded_y_test, decoded_y_pred, labels=labels),
        index=labels,
        columns=labels,
    ))

    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Scaler saved to: {SCALER_FILE}")
    print(f"Label encoder saved to: {ENCODER_FILE}")

if __name__ == "__main__":
    train_lstm()
