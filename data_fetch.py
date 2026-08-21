"""
fetch_data.py
Fetches historical OHLCV data for DSE-listed stocks.

Primary source: bdshare (scrapes dsebd.org). Requires internet access to
dsebd.org, which only works when run OUTSIDE this sandbox (i.e. on your own
machine / any normal internet connection).

Also supports loading from a local CSV if you export data manually from
DSE's data archive (https://www.dsebd.org/data_archive.php) or amarstock.
"""

import pandas as pd
import os

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_from_bdshare(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for a single ticker via bdshare.

    ticker: DSE trading code, e.g. 'GP', 'SQURPHARMA', 'BEXIMCO'
    start_date, end_date: 'YYYY-MM-DD'
    """
    from bdshare import get_hist_data

    df = get_hist_data(start_date, end_date, ticker)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {ticker}. Check the ticker code and date range."
        )

    df = _normalize_columns(df)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    df.to_csv(cache_path, index=False)
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a local CSV. Expects columns that can be mapped to
    date, open, high, low, close, volume (case-insensitive, flexible naming).
    """
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to: date, open, high, low, close, volume."""
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if c in ("date", "trading_date", "trade_date"):
            rename_map[col] = "date"
        elif c in ("open", "openp"):
            rename_map[col] = "open"
        elif c in ("high",):
            rename_map[col] = "high"
        elif c in ("low",):
            rename_map[col] = "low"
        elif c in ("close", "closep", "ltp"):
            rename_map[col] = "close"
        elif c in ("volume", "vol", "trade"):
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def generate_sample_data(days: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic-looking synthetic OHLCV data for testing the
    pipeline when live DSE access isn't available (e.g. in this sandbox).
    Uses a random walk with drift + volatility clustering, loosely mimicking
    real stock behavior.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    returns = rng.normal(0.0003, 0.015, days)
    # volatility clustering
    vol_regime = rng.normal(0, 1, days)
    vol_regime = pd.Series(vol_regime).rolling(10, min_periods=1).mean().values
    returns = returns + vol_regime * 0.005

    price = 100 * (1 + returns).cumprod()
    close = price
    open_ = close * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, days)))
    volume = rng.integers(10000, 500000, days)

    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


if __name__ == "__main__":
    df = generate_sample_data()
    print(df.head())
    print(f"\nGenerated {len(df)} rows of sample data.")
