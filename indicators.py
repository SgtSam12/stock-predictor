"""
indicators.py
Computes standard technical indicators on OHLCV data.
"""

import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, windows=(20, 50, 200)) -> pd.DataFrame:
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
    return df


def add_ema(df: pd.DataFrame, windows=(12, 26)) -> pd.DataFrame:
    for w in windows:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = df["daily_return"].rolling(window).std() * np.sqrt(252)
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full indicator suite on a copy of the dataframe."""
    df = df.copy()
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    return df


def latest_signal_summary(df: pd.DataFrame) -> dict:
    """
    Human-readable summary of the most recent indicator readings, giving a
    simple rule-based read (NOT the ML prediction).
    """
    last = df.iloc[-1]
    signals = {}

    # Trend via moving averages
    if last["close"] > last.get("sma_50", np.nan) > last.get("sma_200", np.nan):
        signals["trend"] = "Bullish (price > SMA50 > SMA200)"
    elif last["close"] < last.get("sma_50", np.nan) < last.get("sma_200", np.nan):
        signals["trend"] = "Bearish (price < SMA50 < SMA200)"
    else:
        signals["trend"] = "Mixed / sideways"

    # RSI
    rsi = last.get("rsi_14", 50)
    if rsi > 70:
        signals["rsi"] = f"Overbought ({rsi:.1f})"
    elif rsi < 30:
        signals["rsi"] = f"Oversold ({rsi:.1f})"
    else:
        signals["rsi"] = f"Neutral ({rsi:.1f})"

    # MACD
    if last["macd"] > last["macd_signal"]:
        signals["macd"] = "Bullish crossover (MACD > signal)"
    else:
        signals["macd"] = "Bearish crossover (MACD < signal)"

    # Bollinger position
    if last["close"] > last["bb_upper"]:
        signals["bollinger"] = "Above upper band (possible overextension)"
    elif last["close"] < last["bb_lower"]:
        signals["bollinger"] = "Below lower band (possible oversold bounce)"
    else:
        signals["bollinger"] = "Within bands (normal range)"

    return signals
