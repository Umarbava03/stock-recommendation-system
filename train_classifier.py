from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

DATA_FILE = Path("data/training_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "best_classifier.joblib"

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
    "macd",
    "bollinger_width",
    "atr_14",
    "volume_spike_ratio",
    "market_return_5",
    "relative_strength_5",
    "avg_sentiment",
    "market_cap",
    "pe_ratio",
    "eps",
]

TARGET_COLUMN = "target_binary"


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    return df


def train_classifier() -> None:
    df = load_data()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df = df.sort_values("as_of_date").reset_index(drop=True)

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    x_train = train_df[FEATURE_COLUMNS]
    x_test = test_df[FEATURE_COLUMNS]

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000)),
            ]
        ),
        "BalancedLogisticRegression_C0.5": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)),
            ]
        ),
        "BalancedLogisticRegression_C1.0": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)),
            ]
        ),
        "BalancedLogisticRegression_C2.0": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", C=2.0)),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features=None,
        ),
        "DepthTunedRandomForestClassifier": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_depth=12,
            min_samples_leaf=2,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "BalancedLinearSVC_C0.25": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=42, C=0.25)),
            ]
        ),
        "BalancedLinearSVC_C0.5": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=42, C=0.5)),
            ]
        ),
        "BalancedLinearSVC_C1.0": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=42, C=1.0)),
            ]
        ),
        "BalancedLinearSVC_C2.0": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=42, C=2.0)),
            ]
        ),
    }

    results = []
    best_model = None
    best_model_name = None
    best_macro_f1 = -1.0
    best_accuracy = -1.0
    best_balanced_accuracy = -1.0
    best_predictions = None

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_accuracy,
            }
        )

        print(f"\n{model_name} Classification Report")
        print(classification_report(y_test, y_pred))

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_accuracy = accuracy
            best_balanced_accuracy = balanced_accuracy
            best_model = model
            best_model_name = model_name
            best_predictions = y_pred

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)

    print("\nClassifier Comparison")
    print(results_df)

    joblib.dump(best_model, MODEL_FILE)
    print(f"\nBest classifier: {best_model_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best macro F1: {best_macro_f1:.4f}")
    print(f"Best balanced accuracy: {best_balanced_accuracy:.4f}")
    if best_predictions is not None:
        labels = sorted(y_test.unique().tolist())
        print("\nBest Classifier Confusion Matrix")
        print(pd.DataFrame(confusion_matrix(y_test, best_predictions, labels=labels), index=labels, columns=labels))
        recall_by_class = {}
        for label in labels:
            label_mask = y_test == label
            recall_by_class[label] = float((best_predictions[label_mask] == label).mean()) if label_mask.any() else 0.0
        print("\nBest Classifier Recall By Class")
        for label, recall in recall_by_class.items():
            print(f"{label}: {recall:.4f}")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_classifier()
