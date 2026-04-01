from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import SYMBOL
from providers.fmp_provider import fetch_historical_prices

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
HISTORICAL_CACHE_DIR = Path("data/historical_prices")
CACHE_EXPIRY_HOURS = 24

def fetch_stock_data(symbol: str = SYMBOL, force_refresh: bool = False) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{symbol}_daily.csv"
    historical_cache_file = HISTORICAL_CACHE_DIR / f"{symbol}.csv"

    if not force_refresh and is_cache_valid(cache_file):
        print("Loading stock data from cache...")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print("Fetching stock data from FMP API...")
    try:
        df = fetch_historical_prices(symbol)
    except Exception:
        if cache_file.exists():
            print("Live stock fetch failed. Using cached stock data...")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if historical_cache_file.exists():
            print("Live stock fetch failed. Using historical price cache...")
            return pd.read_csv(historical_cache_file, index_col=0, parse_dates=True)
        raise

    df.to_csv(cache_file)
    return df


def is_cache_valid(cache_file: Path) -> bool:
    if not cache_file.exists():
        return False

    modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
    return datetime.now() - modified_time < timedelta(hours=CACHE_EXPIRY_HOURS)


def fetch_recent_n_days(symbol: str = SYMBOL, n: int = 20, force_refresh: bool = False) -> pd.DataFrame:
    dt = fetch_stock_data(symbol, force_refresh=force_refresh)
    return dt.tail(n)



def fetch_recent_14_days(symbol: str = SYMBOL, force_refresh: bool = False) -> pd.DataFrame:
    dt = fetch_stock_data(symbol, force_refresh=force_refresh)
    return dt.tail(14)
