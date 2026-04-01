import argparse
import json
from datetime import datetime
from pathlib import Path

from config import SYMBOL
from features import build_market_context, build_price_features
from fundamentals import fetch_fundamentals
from news_sentiment import get_average_sentiment
from predictor import StockPredictor
from stock_data import fetch_recent_n_days

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
LATEST_PREDICTION_FILE = OUTPUT_DIR / "latest_prediction.json"


def save_prediction(symbol: str, features: dict, results) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "features": features,
        "recommendation": results.recommendation,
        "predicted_return_pct": round(results.raw_prediction * 100, 4),
        "confidence_pct": round(results.confidence * 100, 2),
        "model_used": results.model_used,
        "reasons": results.reasons,
        "probabilities": results.probabilities,
    }

    with open(LATEST_PREDICTION_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_prediction_comparison(symbol: str, features: dict, ml_results, lstm_results, classifier_results, rule_results) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "features": features,
        "ml_prediction": {
            "recommendation": ml_results.recommendation,
            "predicted_return_pct": round(ml_results.raw_prediction * 100, 4),
            "confidence_pct": round(ml_results.confidence * 100, 2),
            "model_used": ml_results.model_used,
            "reasons": ml_results.reasons,
            "probabilities": ml_results.probabilities,
        },
        "lstm_prediction": {
            "recommendation": lstm_results.recommendation,
            "predicted_return_pct": round(lstm_results.raw_prediction * 100, 4),
            "confidence_pct": round(lstm_results.confidence * 100, 2),
            "model_used": lstm_results.model_used,
            "reasons": lstm_results.reasons,
            "probabilities": lstm_results.probabilities,
        },
        "classifier_prediction": {
            "recommendation": classifier_results.recommendation,
            "confidence_pct": round(classifier_results.confidence * 100, 2),
            "model_used": classifier_results.model_used,
            "reasons": classifier_results.reasons,
            "probabilities": classifier_results.probabilities,
        },
        "rule_based_prediction": {
            "recommendation": rule_results.recommendation,
            "confidence_pct": round(rule_results.confidence * 100, 2),
            "model_used": rule_results.model_used,
            "reasons": rule_results.reasons,
            "probabilities": rule_results.probabilities,
        },
    }

    with open(LATEST_PREDICTION_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(symbol: str) -> None:
    """
    Main pipeline:
    1. Fetch stock data
    2. Build price-based features
    3. Fetch news sentiment
    4. Combine features
    5. Run prediction
    6. Print results
    """
    ###fetch stock data

    print("Fetching stock data...")
    data = fetch_recent_n_days(symbol, n=60)
    market_data = fetch_recent_n_days("SPY", n=60)

    ###get feature
    print("Fetching features...")
    features = build_price_features(data)
    features.update(build_market_context(data.tail(20), market_data.tail(20)))

    ###get sentiment

    print("Fetching the sentiment score...")
    try:
        avg_sentiment = get_average_sentiment(symbol)
    except Exception as e:
        print(f"Sentiment Analysis failed as {e}")
        avg_sentiment = 0.0

    features["avg_sentiment"] = avg_sentiment

    print("Fetching fundamentals...")
    try:
        fundamentals = fetch_fundamentals(symbol)
    except Exception as e:
        print(f"Fundamentals fetch failed as {e}")
        fundamentals = {
            "market_cap": 0.0,
            "pe_ratio": 0.0,
            "eps": 0.0,
        }

    features.update(fundamentals)


    ###Prediction
    print("getting prediction...")
    predictor = StockPredictor()
    ml_results = predictor.predict_return_model(features)
    lstm_results = predictor.predict_lstm_model(data)
    classifier_results = predictor.predict_class_model(features)
    rule_results = predictor.rule_based_predict(features)
    save_prediction_comparison(symbol, features, ml_results, lstm_results, classifier_results, rule_results)

    ###OUTPUT
    print("\n" + "=" * 50)
    print(f"Stock analysis for {symbol}")
    print("\n" + "=" * 50)

    for key, value in features.items():
        print(f"{key:20}:{value}")

    print("\nML Prediction:\n")
    print(f"Recommendation: {ml_results.recommendation}\n")
    print(f"Predicted 5-day return: {ml_results.raw_prediction * 100:.2f}%\n")
    print(f"Confidence score: {ml_results.confidence * 100:.2f}%\n")
    print("Reasons:")
    for reasons in ml_results.reasons:
        print(f"-{reasons}")

    print("\nLSTM Prediction:\n")
    print(f"Recommendation: {lstm_results.recommendation}\n")
    if lstm_results.model_used == "LSTMClassifier":
        print("Prediction Type: Binary classification\n")
    else:
        print(f"Predicted 5-day return: {lstm_results.raw_prediction * 100:.2f}%\n")
    print(f"Confidence score: {lstm_results.confidence * 100:.2f}%\n")
    print("Reasons:")
    for reasons in lstm_results.reasons:
        print(f"-{reasons}")

    print("\nClassifier Prediction:\n")
    print(f"Recommendation: {classifier_results.recommendation}\n")
    print(f"Confidence score: {classifier_results.confidence * 100:.2f}%\n")
    print("Reasons:")
    for reasons in classifier_results.reasons:
        print(f"-{reasons}")

    print("\nRule-Based Prediction:\n")
    print(f"Recommendation: {rule_results.recommendation}\n")
    print(f"Confidence score: {rule_results.confidence * 100:.2f}%\n")
    print("Reasons:")
    for reasons in rule_results.reasons:
        print(f"-{reasons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Stock ticker symbol")
    args = parser.parse_args()

    main(args.symbol.upper())
