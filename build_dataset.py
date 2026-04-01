from __future__ import annotations

from pathlib import Path

import pandas as pd

from features import build_market_context, build_price_features
from fundamentals import fetch_fundamentals
from providers.fmp_provider import fetch_historical_prices
from news_sentiment import fetch_news, analyse_sentiment


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORICAL_CACHE_DIR = DATA_DIR / "historical_prices"
HISTORICAL_CACHE_DIR.mkdir(exist_ok=True)
WINDOW_SIZE = 20
FORECAST_HORIZON = 5
OUTPUT_FILE = DATA_DIR / "training_data.csv"
SYMBOLS = ["AAPL", "AMZN", "MSFT", "META", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "WMT"]
MARKET_SYMBOL = "SPY"


def load_or_fetch_historical_prices(symbol: str) -> pd.DataFrame:
    cache_file = HISTORICAL_CACHE_DIR / f"{symbol}.csv"

    try:
        df = fetch_historical_prices(symbol)
        df.to_csv(cache_file)
        return df
    except Exception:
        if cache_file.exists():
            print(f"Live price fetch failed for {symbol}. Using cached historical prices...")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise


def build_training_dataset(symbol: str) -> pd.DataFrame:
    df = load_or_fetch_historical_prices(symbol)
    market_df = load_or_fetch_historical_prices(MARKET_SYMBOL)
    fundamentals = fetch_fundamentals(symbol)

    sentiment_df = get_historical_sentiment(symbol)

    rows = []

    for i in range(WINDOW_SIZE - 1, len(df) - FORECAST_HORIZON):
        window = df.iloc[i - WINDOW_SIZE + 1 : i + 1]
        features = build_price_features(window)
        market_window = market_df.loc[market_df.index <= window.index[-1]].tail(WINDOW_SIZE)
        features.update(build_market_context(window, market_window))

        current_close = float(window["close"].iloc[-1])
        future_close = float(df["close"].iloc[i + FORECAST_HORIZON])
        future_returns = (future_close - current_close) / current_close

        window_return= window["close"].pct_change().dropna()
        volatility_threshold=float(window_return.std()) if not window_return.empty else 0.0

        threshold=max(volatility_threshold, 0.01)

        as_of_date = pd.Timestamp(window.index[-1])
        avg_sentiment = get_sentiment_as_of(sentiment_df, as_of_date)


        if future_returns > threshold:
            target_class = "BUY"
        elif future_returns < -threshold:
            target_class = "NOT_BUY"
        else:
            target_class = "HOLD"

        target_binary = "BUY" if future_returns > threshold else "NOT_BUY"

        row = {
            "symbol": symbol,
            "as_of_date": str(window.index[-1].date()),
            **features,
            "avg_sentiment": avg_sentiment,
            "market_cap": fundamentals["market_cap"],
            "pe_ratio": fundamentals["pe_ratio"],
            "eps": fundamentals["eps"],
            "current_close": current_close,
            "future_close": future_close,
            "target_return": future_returns,
            "target_class": target_class,
            "target_binary": target_binary,
        }
        rows.append(row)

    return pd.DataFrame(rows)

def get_historical_sentiment(symbol: str) -> pd.DataFrame:
    try:
        news_df = fetch_news(symbol)
        sentiment_df = analyse_sentiment(news_df, symbol)

        if sentiment_df.empty:
            return pd.DataFrame()

        sentiment_df["time_published"] = pd.to_datetime(
            sentiment_df["time_published"], errors="coerce"
        )

        sentiment_df = sentiment_df.dropna(subset=["time_published"])
        return sentiment_df.sort_values("time_published").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_sentiment_as_of(
    sentiment_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int = 7,
) -> float:
    if sentiment_df.empty:
        return 0.0

    start_date = as_of_date - pd.Timedelta(days=lookback_days)

    window_df = sentiment_df[
        (sentiment_df["time_published"] <= as_of_date)
        & (sentiment_df["time_published"] >= start_date)
    ]

    if window_df.empty:
        return 0.0

    total_weight = window_df["ticker_relevance"].sum()
    if total_weight == 0:
        return 0.0

    return float(window_df["weighted_sentiment"].sum() / total_weight)

def main():
    datasets = []
    for symbol in SYMBOLS:
        print(f"Building dataset for {symbol}...")
        try:
            dataset = build_training_dataset(symbol)
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")
            continue
        if not dataset.empty:
            datasets.append(dataset)

    if not datasets:
        raise RuntimeError("No datasets were built successfully.")
    dataset = pd.concat(datasets, ignore_index=True)
    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Shape: {dataset.shape}")
    print(dataset.head())




if __name__ == "__main__":
    main()
