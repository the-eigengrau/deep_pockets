"""
Utilities for basic price feature engineering.

All helpers operate per-ticker and use only trailing windows (no look-ahead).
"""

from typing import Optional

import pandas as pd  # type: ignore


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by ticker/date to make groupby operations predictable."""
    if not {"ticker", "date"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'ticker' and 'date' columns.")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_returns(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    col_name: str = "ret_1d",
) -> pd.DataFrame:
    """
    Add 1-day returns per ticker based on `price_col`.

    ret_1d_t = price_t / price_{t-1} - 1
    """
    df = _ensure_sorted(df)
    if price_col not in df.columns:
        raise ValueError(f"Expected column '{price_col}' in prices DataFrame.")

    df[col_name] = (
        df.groupby("ticker", observed=True)[price_col]
        .pct_change(1)
        .astype("float64")
    )
    return df


def rolling_momentum(
    df: pd.DataFrame,
    window: int,
    price_col: str = "adj_close",
    col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add N-day price momentum per ticker:

    momN_t = price_t / price_{t-N} - 1
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    df = _ensure_sorted(df)
    if price_col not in df.columns:
        raise ValueError(f"Expected column '{price_col}' in prices DataFrame.")

    out_col = col_name or f"mom{window}"
    df[out_col] = (
        df.groupby("ticker", observed=True)[price_col]
        .pct_change(window)
        .astype("float64")
    )
    return df


def rolling_volatility(
    df: pd.DataFrame,
    window: int,
    ret_col: str = "ret_1d",
    col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add rolling volatility of 1-day returns per ticker:

    volN_t = std(ret_{t-N+1}, ..., ret_t)
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    df = _ensure_sorted(df)
    if ret_col not in df.columns:
        raise ValueError(f"Expected column '{ret_col}' in DataFrame.")

    out_col = col_name or f"vol{window}"
    df[out_col] = (
        df.groupby("ticker", observed=True)[ret_col]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
        .astype("float64")
    )
    return df


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score using only trailing data:

    z_t = (x_t - mean_{t-window+1:t}) / std_{t-window+1:t}
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    roll = series.rolling(window)
    mean = roll.mean()
    std = roll.std()
    z = (series - mean) / std
    return z.astype("float64")
