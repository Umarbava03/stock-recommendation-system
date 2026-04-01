import pandas as pd
import requests

from config import ALPHA_VANTAGE_API_KEY


def fetch_news_feed(symbol: str, limit: int = 10) -> pd.DataFrame:
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("Missing Alpha Vantage API key.")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "limit": limit,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "feed" not in data:
        raise ValueError(f"Alpha Vantage API error: {data}")

    rows = []
    for item in data["feed"]:
        rows.append(
            {
                "time_published": item.get("time_published"),
                "source": item.get("source"),
                "summary": item.get("summary"),
                "headline": item.get("title"),
                "ticker_sentiment": item.get("ticker_sentiment", []),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["time_published"] = pd.to_datetime(
            df["time_published"],
            format="%Y%m%dT%H%M%S",
            errors="coerce",
        )

    return df
