from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from features import build_price_features

MODEL_PATH = "models/best_regressor.joblib"
CLASSIFIER_MODEL_PATH = "models/best_classifier.joblib"
LSTM_MODEL_PATH = "models/best_lstm_classifier.keras"
LSTM_SCALER_PATH = "models/best_lstm_classifier_scaler.joblib"
LSTM_LABEL_ENCODER_PATH = "models/best_lstm_label_encoder.joblib"
LSTM_SEQUENCE_LENGTH = 20


@dataclass
class PredictionResult:
    recommendation: str
    confidence: float
    model_used: str
    reasons: List[str]
    raw_prediction: float
    probabilities: Dict[str, float]


class StockPredictor:
    """
    Stock recommendation predictor with both rule-based and regressor-based paths.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        classifier_model_path: str = CLASSIFIER_MODEL_PATH,
        lstm_model_path: str = LSTM_MODEL_PATH,
        lstm_scaler_path: str = LSTM_SCALER_PATH,
        lstm_label_encoder_path: str = LSTM_LABEL_ENCODER_PATH,
    ) -> None:
        self.model_path = model_path
        self.classifier_model_path = classifier_model_path
        self.lstm_model_path = lstm_model_path
        self.lstm_scaler_path = lstm_scaler_path
        self.lstm_label_encoder_path = lstm_label_encoder_path
        self.model: Optional[RandomForestRegressor] = None
        self.classifier_model = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_label_encoder = None
        self.feature_columns: List[str] = [
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
        self.lstm_feature_columns: List[str] = [
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

    def rule_based_predict(self, features: Dict[str, float]) -> PredictionResult:
        price_change = float(features.get("price_change_pct", 0.0))
        volatility = float(features.get("volatility", 0.0))
        trend_signal = int(features.get("trend_signal", 0))
        avg_sentiment = float(features.get("avg_sentiment", 0.0))
        pe_ratio = float(features.get("pe_ratio", 0.0))
        eps = float(features.get("eps", 0.0))

        score = 0
        reasons: List[str] = []

        if price_change > 2:
            score += 1
            reasons.append("The stock has shown positive price growth over the last 14 trading days.")
        elif price_change < -2:
            score -= 1
            reasons.append("The stock has shown negative price movement over the last 14 trading days.")

        if trend_signal == 1:
            score += 1
            reasons.append("The short-term moving average is above the long-term moving average, indicating an upward trend.")
        else:
            score -= 1
            reasons.append("The short-term moving average is not above the long-term moving average, indicating weak momentum.")

        if avg_sentiment > 0.15:
            score += 1
            reasons.append("Recent news sentiment is positive.")
        elif avg_sentiment < -0.15:
            score -= 1
            reasons.append("Recent news sentiment is negative.")
        else:
            reasons.append("Recent news sentiment is neutral.")

        if 0 < pe_ratio < 35:
            score += 1
            reasons.append("The P/E ratio is within a generally acceptable range.")
        elif pe_ratio >= 35:
            score -= 1
            reasons.append("The P/E ratio is relatively high, which may indicate overvaluation.")

        if eps > 0:
            score += 1
            reasons.append("The company has positive earnings per share.")
        else:
            score -= 1
            reasons.append("The company has weak or negative earnings per share.")

        if volatility > 0.04:
            score -= 1
            reasons.append("The stock shows relatively high volatility, which increases short-term risk.")
        else:
            reasons.append("The stock volatility is within a moderate range.")

        if score >= 3:
            recommendation = "BUY"
            raw_prediction = 1
            confidence = min(0.95, 0.55 + (score * 0.08))
        elif score <= 0:
            recommendation = "NOT BUY"
            raw_prediction = 0
            confidence = min(0.95, 0.55 + (abs(score) * 0.08))
        else:
            recommendation = "HOLD / WATCH"
            raw_prediction = 0
            confidence = 0.55
        if recommendation == "BUY":
            buy_prob = confidence
            not_buy_prob = 1 - confidence
        elif recommendation == "NOT BUY":
            buy_prob = 1 - confidence
            not_buy_prob = confidence
        else:  # HOLD
            buy_prob = 0.5
            not_buy_prob = 0.5

        probabilities = {
            "BUY": round(buy_prob, 4),
            "NOT_BUY": round(not_buy_prob, 4),
        }
        
        return PredictionResult(
            recommendation=recommendation,
            confidence=round(confidence, 4),
            model_used="rule_based",
            reasons=reasons,
            raw_prediction=float(raw_prediction),
            probabilities=probabilities,
        )

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model = joblib.load(self.model_path)

    def load_classifier_model(self) -> None:
        if not os.path.exists(self.classifier_model_path):
            raise FileNotFoundError(f"Classifier model not found at {self.classifier_model_path}")

        self.classifier_model = joblib.load(self.classifier_model_path)

    def load_lstm_model(self) -> None:
        if not os.path.exists(self.lstm_model_path):
            raise FileNotFoundError(f"LSTM model not found at {self.lstm_model_path}")
        if not os.path.exists(self.lstm_scaler_path):
            raise FileNotFoundError(f"LSTM scaler not found at {self.lstm_scaler_path}")
        if not os.path.exists(self.lstm_label_encoder_path):
            raise FileNotFoundError(f"LSTM label encoder not found at {self.lstm_label_encoder_path}")

        from tensorflow.keras.models import load_model

        self.lstm_model = load_model(self.lstm_model_path)
        self.lstm_scaler = joblib.load(self.lstm_scaler_path)
        self.lstm_label_encoder = joblib.load(self.lstm_label_encoder_path)

    def predict_return_model(self, feature: Dict[str, float]) -> PredictionResult:
        if self.model is None:
            self.load_model()

        input_row = {col: float(feature.get(col, 0.0)) for col in self.feature_columns}
        x_input = pd.DataFrame([input_row])

        predicted_return = float(self.model.predict(x_input)[0])
        reasons: List[str] = [
            f"Predicted 5-day return: {predicted_return * 100:.2f}%."
        ]

        if predicted_return > 0.01:
            recommendation = "BUY"
            confidence = min(0.95, 0.55 + abs(predicted_return) * 5)
            reasons.append("The predicted return is meaningfully positive.")
        elif predicted_return < -0.02:
            recommendation = "NOT BUY"
            confidence = min(0.95, 0.55 + abs(predicted_return) * 5)
            reasons.append("The predicted return is meaningfully negative.")

        else:
            recommendation = "HOLD / WATCH"
            confidence = 0.55 + min(0.10, abs(predicted_return) * 2)
            reasons.append("The predicted return is too small to justify a strong action.")

        probabilities = {
            "BUY": round(max(predicted_return, 0.0), 4),
            "NOT_BUY": round(max(-predicted_return, 0.0), 4),
        }

        return PredictionResult(
            recommendation=recommendation,
            confidence=round(confidence, 4),
            model_used=type(self.model).__name__,
            reasons=reasons,
            raw_prediction=predicted_return,
            probabilities=probabilities,
        )

    def predict_class_model(self, feature: Dict[str, float]) -> PredictionResult:
        if self.classifier_model is None:
            self.load_classifier_model()

        input_row = {col: float(feature.get(col, 0.0)) for col in self.feature_columns}
        x_input = pd.DataFrame([input_row])

        predicted_class = str(self.classifier_model.predict(x_input)[0])
        probabilities = {}

        if hasattr(self.classifier_model, "predict_proba"):
            class_labels = list(self.classifier_model.classes_)
            class_probs = self.classifier_model.predict_proba(x_input)[0]
            probabilities = {
                str(label): round(float(prob), 4)
                for label, prob in zip(class_labels, class_probs)
            }
            confidence = max(class_probs)
        else:
            confidence = 0.55

        class_set = set(map(str, getattr(self.classifier_model, "classes_", [])))

        if predicted_class == "BUY":
            if class_set == {"BUY", "NOT_BUY"}:
                reasons = ["The binary classifier predicts upside above the risk-adjusted threshold."]
            else:
                reasons = ["The classifier predicts a positive actionable move."]
        elif predicted_class == "NOT_BUY":
            if class_set == {"BUY", "NOT_BUY"}:
                reasons = ["The binary classifier does not see enough upside above the risk-adjusted threshold."]
            else:
                reasons = ["The classifier predicts a negative actionable move."]
        else:
            reasons = ["The classifier does not detect a strong enough edge."]

        return PredictionResult(
            recommendation=str(predicted_class).replace("_", " "),
            confidence=round(float(confidence), 4),
            model_used=type(self.classifier_model).__name__,
            reasons=reasons,
            raw_prediction=0.0,
            probabilities=probabilities,
        )

    def predict_lstm_model(
        self,
        recent_price_data: pd.DataFrame,
    ) -> PredictionResult:
        if self.lstm_model is None or self.lstm_scaler is None:
            self.load_lstm_model()

        required_points = LSTM_SEQUENCE_LENGTH + 20 - 1
        if len(recent_price_data) < required_points:
            raise ValueError(
                f"LSTM prediction requires at least {required_points} rows of price data."
            )

        sequence_rows = []
        for end_idx in range(20 - 1, len(recent_price_data)):
            window = recent_price_data.iloc[end_idx - 20 + 1 : end_idx + 1]
            row_features = build_price_features(window)
            sequence_rows.append(
                [float(row_features.get(col, 0.0)) for col in self.lstm_feature_columns]
            )

        if len(sequence_rows) < LSTM_SEQUENCE_LENGTH:
            raise ValueError("Not enough feature windows to build LSTM sequence.")

        latest_sequence = np.array(sequence_rows[-LSTM_SEQUENCE_LENGTH:], dtype=float)
        sequence_df = pd.DataFrame(latest_sequence, columns=self.lstm_feature_columns)
        scaled_sequence = self.lstm_scaler.transform(sequence_df)
        x_input = np.expand_dims(scaled_sequence, axis=0)
        class_probs = self.lstm_model.predict(x_input, verbose=0)[0]
        predicted_index = int(np.argmax(class_probs))
        predicted_class = str(self.lstm_label_encoder.inverse_transform([predicted_index])[0])
        confidence = float(np.max(class_probs))

        if predicted_class == "BUY":
            recommendation = "BUY"
            reasons = ["The binary LSTM classifier predicts upside above the risk-adjusted threshold."]
        else:
            recommendation = "NOT BUY"
            reasons = ["The binary LSTM classifier does not see enough upside above the risk-adjusted threshold."]

        probabilities = {
            str(label): round(float(prob), 4)
            for label, prob in zip(self.lstm_label_encoder.classes_, class_probs)
        }

        return PredictionResult(
            recommendation=recommendation,
            confidence=round(confidence, 4),
            model_used="LSTMClassifier",
            reasons=reasons,
            raw_prediction=0.0,
            probabilities=probabilities,
        )
