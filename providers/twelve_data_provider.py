from __future__ import annotations

import pandas as pd
import requests

from config import TWELVE_DATA_API_KEY


# ---------------------------------------------------------------------
# Fetch Daily Time Series Data
# ---------------------------------------------------------------------
def fetch_daily_time_series(symbol: str, outputsize: int = 30) -> pd.DataFrame:
    """
    Fetch daily OHLCV stock data from Twelve Data API.

    Args:
        symbol (str): Stock symbol (e.g., AAPL)
        outputsize (int): Number of days to fetch

    Returns:
        pd.DataFrame with columns:
        ['open', 'high', 'low', 'close', 'volume']
    """

    if not TWELVE_DATA_API_KEY:
        raise ValueError("❌ Missing Twelve Data API key.")

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"❌ API request failed: {e}")

    # ------------------------------------------------------------------
    # Validate response
    # ------------------------------------------------------------------
    if "values" not in data:
        raise ValueError(f"❌ Twelve Data API error: {data}")

    # ------------------------------------------------------------------
    # Convert to DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(data["values"])

    if df.empty:
        raise ValueError("❌ Empty data received from API.")
    print("Rows returned by Twelve Data:", len(df))


    # ------------------------------------------------------------------
    # Data Cleaning
    # ------------------------------------------------------------------
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df = df.set_index("datetime")
    df = df.sort_index()  # oldest → newest

    # Convert columns to numeric
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    return df[numeric_cols]
