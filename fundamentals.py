from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from config import SYMBOL
from providers.fmp_provider import fetch_company_profile, fetch_ratios_ttm

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
FUNDAMENTALS_CACHE_EXPIRY_HOURS = 24
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def is_fundamentals_cache_valid(cache_file: Path) -> bool:
    if not cache_file.exists():
        return False

    modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
    return datetime.now() - modified_time < timedelta(hours=FUNDAMENTALS_CACHE_EXPIRY_HOURS)


def load_fundamentals_cache(cache_file: Path) -> dict:
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fundamentals_cache(data: dict, cache_file: Path) -> None:
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def fetch_fundamentals(symbol: str = SYMBOL, force_refresh: bool = False) -> dict:
    cache_file = CACHE_DIR / f"{symbol}_fundamentals.json"

    if not force_refresh and is_fundamentals_cache_valid(cache_file):
        print("Loading fundamentals from cache...")
        return load_fundamentals_cache(cache_file)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching fundamentals from FMP API... (attempt {attempt})")

            profile = fetch_company_profile(symbol)
            ratios = fetch_ratios_ttm(symbol)

            fundamentals = {
                "market_cap": float(profile.get("marketCap", 0.0)),
                "pe_ratio": float(ratios.get("priceToEarningsRatioTTM", 0.0)),
                "eps": float(ratios.get("netIncomePerShareTTM", 0.0)),
            }

            save_fundamentals_cache(fundamentals, cache_file)
            return fundamentals

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} failed: {e}")

            if attempt < MAX_RETRIES:
                print("Retrying...")
                time.sleep(RETRY_DELAY_SECONDS)

    if cache_file.exists():
        print("API failed. Using cached fundamentals...")
        return load_fundamentals_cache(cache_file)

    raise RuntimeError(f"Failed to fetch fundamentals: {last_error}")
