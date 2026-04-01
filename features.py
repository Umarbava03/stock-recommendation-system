import pandas as pd


def build_price_features(df: pd.DataFrame) -> dict:
    recent = df.copy()

    recent["ma_5"] = recent["close"].rolling(5).mean()
    recent["ma_10"] = recent["close"].rolling(10).mean()
    recent["ma_20"] = recent["close"].rolling(20).mean()
    recent["daily_return"] = recent["close"].pct_change()
    recent["momentum_5"] = recent["close"] - recent["close"].shift(5)
    recent["volatility_5"] = recent["daily_return"].rolling(5).std()
    recent["price_vs_ma_10"] = (recent["close"] - recent["ma_10"]) / recent["ma_10"]
    recent["ema_12"] = recent["close"].ewm(span=12, adjust=False).mean()
    recent["ema_26"] = recent["close"].ewm(span=26, adjust=False).mean()
    recent["macd"] = recent["ema_12"] - recent["ema_26"]

    recent["bb_std_20"] = recent["close"].rolling(20).std()
    recent["bollinger_width"] = (2 * recent["bb_std_20"] * 2) / recent["ma_20"]

    prev_close = recent["close"].shift(1)
    true_range = pd.concat(
        [
            recent["high"] - recent["low"],
            (recent["high"] - prev_close).abs(),
            (recent["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    recent["atr_14"] = true_range.rolling(14).mean()

    recent["volume_avg_20"] = recent["volume"].rolling(20).mean()
    recent["volume_spike_ratio"] = recent["volume"] / recent["volume_avg_20"]

    delta = recent["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    recent["rsi_14"] = 100 - (100 / (1 + rs))

    price_change_pct = (
        (recent["close"].iloc[-1] - recent["close"].iloc[0])
        / recent["close"].iloc[0]
    ) * 100

    volatility = recent["daily_return"].std()
    avg_volume = recent["volume"].mean()

    ma_5 = recent["ma_5"].iloc[-1]
    ma_10 = recent["ma_10"].iloc[-1]

    trend_signal = 1 if ma_5 > ma_10 else 0
    latest = recent.iloc[-1]

    return {
        "price_change_pct": float(price_change_pct),
        "volatility": float(volatility) if pd.notnull(volatility) else 0.0,
        "avg_volume": float(avg_volume),
        "ma_5": float(ma_5) if pd.notnull(ma_5) else 0.0,
        "ma_10": float(ma_10) if pd.notnull(ma_10) else 0.0,
        "trend_signal": trend_signal,
        "ma_20": float(latest["ma_20"]) if pd.notnull(latest["ma_20"]) else 0.0,
        "momentum_5": float(latest["momentum_5"]) if pd.notnull(latest["momentum_5"]) else 0.0,
        "volatility_5": float(latest["volatility_5"]) if pd.notnull(latest["volatility_5"]) else 0.0,
        "price_vs_ma_10": float(latest["price_vs_ma_10"]) if pd.notnull(latest["price_vs_ma_10"]) else 0.0,
        "rsi_14": float(latest["rsi_14"]) if pd.notnull(latest["rsi_14"]) else 0.0,
        "macd": float(latest["macd"]) if pd.notnull(latest["macd"]) else 0.0,
        "bollinger_width": float(latest["bollinger_width"]) if pd.notnull(latest["bollinger_width"]) else 0.0,
        "atr_14": float(latest["atr_14"]) if pd.notnull(latest["atr_14"]) else 0.0,
        "volume_spike_ratio": float(latest["volume_spike_ratio"]) if pd.notnull(latest["volume_spike_ratio"]) else 0.0,
    }


def build_market_context(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> dict:
    if len(stock_df) < 6 or len(market_df) < 6:
        return {
            "market_return_5": 0.0,
            "relative_strength_5": 0.0,
        }

    stock_return_5 = (
        stock_df["close"].iloc[-1] - stock_df["close"].iloc[-6]
    ) / stock_df["close"].iloc[-6]
    market_return_5 = (
        market_df["close"].iloc[-1] - market_df["close"].iloc[-6]
    ) / market_df["close"].iloc[-6]

    return {
        "market_return_5": float(market_return_5),
        "relative_strength_5": float(stock_return_5 - market_return_5),
    }
