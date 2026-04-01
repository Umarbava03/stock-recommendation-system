from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import SYMBOL
from providers.alpha_vantage_provider import fetch_news_feed
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
NEWS_CACHE_EXPIRY_HOURS = 6

try:
    nltk.data.find("sentiment/vader_lexicon.zip")

except LookupError:
    nltk.download("vader_lexicon")

sentiment_analyzer = SentimentIntensityAnalyzer()



def is_news_cache_valid(cache_file: Path) -> bool:
    if not cache_file.exists():
        return False

    modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
    return datetime.now() - modified_time < timedelta(hours=NEWS_CACHE_EXPIRY_HOURS)


def load_news_cache(cache_file: Path) -> pd.DataFrame:
    with open(cache_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    if not df.empty and "time_published" in df.columns:
        df["time_published"] = pd.to_datetime(df["time_published"], errors="coerce")

    return df


def save_news_cache(df: pd.DataFrame, cache_file: Path) -> None:
    records = df.copy()

    if not records.empty and "time_published" in records.columns:
        records["time_published"] = records["time_published"].astype(str)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(records.to_dict(orient="records"), f, indent=2)


def fetch_news(symbol: str = SYMBOL, limit: int = 10, force_refresh: bool = False) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{symbol}_news.json"

    if not force_refresh and is_news_cache_valid(cache_file):
        print("Loading news data from cache...")
        return load_news_cache(cache_file)

    try:
        print("Fetching news data from Alpha Vantage API...")
        df = fetch_news_feed(symbol, limit)
        save_news_cache(df, cache_file)
        return df
    except Exception:
        if cache_file.exists():
            print("Live news fetch failed. Using cached news data...")
            return load_news_cache(cache_file)
        raise


def get_ticker_relevance(ticker_sentiment_list, symbol: str = SYMBOL) -> float:
    if not isinstance(ticker_sentiment_list, list):
        return 0.0

    for item in ticker_sentiment_list:
        if item.get("ticker") == symbol:
            return float(item.get("relevance_score", 0.0))
    return 0.0


def analyse_sentiment(df_news: pd.DataFrame, symbol: str = SYMBOL) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame()

    rows = []
    

    for _, row in df_news.iterrows():
        relevance = get_ticker_relevance(row.get("ticker_sentiment"), symbol)
        headline = row.get("headline") or ""
        summary = row.get("summary") or ""
        article = f"{headline}".strip()

    

        if relevance < 0.7:
            continue

        score = sentiment_analyzer.polarity_scores(article)

        p_positive = score.get("pos", 0.0)
        p_negative = score.get("neg", 0.0)
        p_neutral = score.get("neu", 0.0)
        sentiment_score = score.get("compound", 0.0)
        weighted_sentiment = sentiment_score * relevance

        

        rows.append(
            {
                "time_published": row.get("time_published"),
                "source": row.get("source"),
                "headline": headline,
                "summary": summary,
                "article_text": article,
                "ticker_relevance": relevance,
                "p_positive": p_positive,
                "p_negative": p_negative,
                "p_neutral": p_neutral,
                "sentiment_score": sentiment_score,
                "weighted_sentiment": weighted_sentiment,
            }
        )

    
    return pd.DataFrame(rows)


def get_average_sentiment(symbol: str = SYMBOL, force_refresh: bool = False) -> float:
    try:
        df_news = fetch_news(symbol, force_refresh=force_refresh)
    except Exception:
        return 0.0

    if df_news.empty:
        return 0.0

    try:
        df_sentiment = analyse_sentiment(df_news, symbol)
    except Exception:
        return 0.0

    if df_sentiment.empty:
        return 0.0

    total_weight = df_sentiment["ticker_relevance"].sum()
    if total_weight == 0:
        return 0.0
    


    return float(df_sentiment["weighted_sentiment"].sum() / total_weight)
