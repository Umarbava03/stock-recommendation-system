import requests
import pandas as pd
from config import FMP_API_KEY

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

def fetch_company_profile(symbol: str) -> dict:
    if not FMP_API_KEY:
        raise ValueError("Missing FMP API key.")

    url = "https://financialmodelingprep.com/stable/profile"
    params = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP profile API error: {data}")

    return data[0]


def fetch_company_quote(symbol: str) -> dict:
    if not FMP_API_KEY:
        raise ValueError("Missing FMP API key.")

    url = "https://financialmodelingprep.com/stable/quote"
    params = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP quote API error: {data}")

    return data[0]


def _get_fmp_data(url: str, symbol: str):
    if not FMP_API_KEY:
        raise ValueError("Missing FMP API key.")

    params = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
    }

    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP API error: {data}")

    return data

def fetch_ratios_ttm(symbol: str) -> dict:
    if not FMP_API_KEY:
        raise ValueError("Missing FMP API key.")

    url = "https://financialmodelingprep.com/stable/ratios-ttm"
    params = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP ratios TTM API error: {data}")

    return data[0]


def fetch_key_metrics_ttm(symbol: str) -> dict:
    if not FMP_API_KEY:
        raise ValueError("Missing FMP API key.")

    url = "https://financialmodelingprep.com/stable/key-metrics-ttm"
    params = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP key metrics TTM API error: {data}")

    return data[0]

def fetch_historical_prices(symbol: str) -> pd.DataFrame:
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    data = _get_fmp_data(url, symbol)

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError(f"No historical price data returned for {symbol}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date")
    df = df.sort_index()

    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }

    df = df[list(rename_map.keys())]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    return df
