"""
Stub: Build features from prices and text sentiment.

Pipeline (to implement later):
1) Load price parquet from data/prices/prices.parquet.
2) Compute features per ticker/day: ret_1d, mom5, mom20, vol5.
3) Load headlines from data/text/headlines_raw.parquet.
4) Run FinBERT (yiyanghkust/finbert-tone) to get pos/neu/neg probabilities per row.
5) Aggregate to daily per (date, ticker): mean probs, senti_count.
6) Apply 1-day lag to daily sentiment features before merging.
7) Rolling z-score each price feature per ticker (no leakage).
8) Outer join and fill missing sentiment with 0; write features/features.parquet.

Outputs columns:
[date, ticker, mom5, mom20, vol5, senti_pos, senti_neg, senti_neu, senti_count]
"""

from pathlib import Path

import pandas as pd  # type: ignore
import yaml  # type: ignore

from .utils import compute_returns, rolling_momentum, rolling_volatility, rolling_zscore


def project_root() -> Path:
    """Return repository root (one level up from this file)."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """
    Minimal price feature pipeline.

    Steps:
    1) Load prices from data/prices/prices.parquet.
    2) Compute per-ticker daily features: ret_1d, mom5, mom20, vol5.
    3) Apply rolling per-ticker z-scores with window from conf.yaml.
    4) Write [date, ticker, ret_1d, mom5, mom20, vol5] to features/features.parquet.
    """
    root = project_root()

    # Load config
    conf_path = root / "conf.yaml"
    with conf_path.open("r") as f:
        conf = yaml.safe_load(f)

    feat_conf = conf.get("features", {}) if isinstance(conf, dict) else {}
    zscore_window = int(feat_conf.get("zscore_window", 252))
    vol_window = int(feat_conf.get("vol_window", 5))

    # Load raw prices
    prices_path = root / "data" / "prices" / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices parquet not found at {prices_path}")

    df = pd.read_parquet(prices_path)
    if df.empty:
        raise ValueError(f"No rows found in {prices_path}")

    # Ensure basic schema
    required_cols = {"date", "ticker", "adj_close"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in prices parquet: {missing}")

    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.dropna(subset=["date"])

    # Sort once for stability
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 1-day returns
    df = compute_returns(df, price_col="adj_close", col_name="ret_1d")

    # 5-day and 20-day momentum from prices
    df = rolling_momentum(df, window=5, price_col="adj_close", col_name="mom5")
    df = rolling_momentum(df, window=20, price_col="adj_close", col_name="mom20")

    # 5-day volatility of 1-day returns
    df = rolling_volatility(df, window=vol_window, ret_col="ret_1d", col_name=f"vol{vol_window}")

    # Rolling per-ticker z-scores for each feature (in-place overwrite)
    feat_cols = ["ret_1d", "mom5", "mom20", f"vol{vol_window}"]
    for col in feat_cols:
        df[col] = df.groupby("ticker", observed=True)[col].transform(
            lambda s: rolling_zscore(s, zscore_window)
        )

    # Drop rows without fully-formed features
    df = df.dropna(subset=feat_cols)

    # Final column ordering (simple price-only table for now)
    out_cols = ["date", "ticker"] + feat_cols
    df_out = df[out_cols].sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path = root / "features" / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"Wrote {len(df_out):,} feature rows to {out_path}")


if __name__ == "__main__":
    main()


