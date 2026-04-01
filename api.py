from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
import pandas as pd

from fastapi import FastAPI, HTTPException, Query

from features import build_market_context, build_price_features
from fundamentals import fetch_fundamentals
from news_sentiment import get_average_sentiment
from predictor import StockPredictor
from stock_data import fetch_recent_n_days


app = FastAPI(
    title="Stock Recommendation API",
    version="1.0.0",
    description="API for stock recommendation predictions using the project models.",
)


def build_price_history(data: pd.DataFrame, rows: int = 60) -> list[dict[str, Any]]:
    chart_df = data.tail(rows).copy()
    chart_df["ma_5"] = chart_df["close"].rolling(5).mean()
    chart_df["ma_10"] = chart_df["close"].rolling(10).mean()
    chart_df["ma_20"] = chart_df["close"].rolling(20).mean()

    chart_df = chart_df.reset_index()
    date_column = chart_df.columns[0]

    return [
        {
            "date": pd.Timestamp(row[date_column]).date().isoformat(),
            "close": round(float(row["close"]), 4),
            "ma_5": round(float(row["ma_5"]), 4) if pd.notna(row["ma_5"]) else None,
            "ma_10": round(float(row["ma_10"]), 4) if pd.notna(row["ma_10"]) else None,
            "ma_20": round(float(row["ma_20"]), 4) if pd.notna(row["ma_20"]) else None,
        }
        for _, row in chart_df.iterrows()
    ]


def run_prediction(symbol: str) -> Dict[str, Any]:
    symbol = symbol.upper()

    data = fetch_recent_n_days(symbol, n=60)
    if len(data) < 49:
        historical_df = pd.read_csv(
            f"data/historical_prices/{symbol}.csv",
            index_col=0,
            parse_dates=True,
        )
        data = historical_df.tail(60)

    market_data = fetch_recent_n_days("SPY", n=60)
    if len(market_data) < 20:
        historical_market_df = pd.read_csv(
            "data/historical_prices/SPY.csv",
            index_col=0,
            parse_dates=True,
        )
        market_data = historical_market_df.tail(60)

    features = build_price_features(data)
    features.update(build_market_context(data.tail(20), market_data.tail(20)))

    try:
        avg_sentiment = get_average_sentiment(symbol)
    except Exception:
        avg_sentiment = 0.0
    features["avg_sentiment"] = avg_sentiment

    try:
        fundamentals = fetch_fundamentals(symbol)
    except Exception:
        fundamentals = {
            "market_cap": 0.0,
            "pe_ratio": 0.0,
            "eps": 0.0,
        }
    features.update(fundamentals)

    predictor = StockPredictor()
    ml_results = predictor.predict_return_model(features)
    lstm_results = predictor.predict_lstm_model(data)
    classifier_results = predictor.predict_class_model(features)
    rule_results = predictor.rule_based_predict(features)

    return {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "features": features,
        "price_history": build_price_history(data),
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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/predict")
def predict(symbol: str = Query(..., min_length=1, max_length=10)) -> Dict[str, Any]:
    try:
        return run_prediction(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
