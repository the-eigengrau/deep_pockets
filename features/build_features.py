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
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
import yaml  # type: ignore

from .utils import compute_returns, rolling_momentum, rolling_volatility, rolling_zscore


def project_root() -> Path:
    """Return repository root (one level up from this file)."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """
    Build fused daily features from prices and text sentiment.

    Steps:
    1) Load prices from data/prices/prices.parquet.
    2) Compute per-ticker daily price features: ret_1d, mom5, mom20, vol5.
    3) Apply rolling per-ticker z-scores with window from conf.yaml.
    4) Load sentiment CSVs from data/text/, pick the one with more (date, ticker) rows.
    5) Clean sentiment dates/tickers, enforce unique (date, ticker).
    6) Optionally apply rolling per-ticker z-scores to sentiment features.
    7) Left-join price and sentiment on (date, ticker), neutral-fill missing sentiment.
    8) Write fused table to features/features.parquet.
    """
    root = project_root()

    # Load config
    conf_path = root / "conf.yaml"
    with conf_path.open("r") as f:
        conf = yaml.safe_load(f)

    feat_conf = conf.get("features", {}) if isinstance(conf, dict) else {}
    zscore_window = int(feat_conf.get("zscore_window", 252))
    vol_window = int(feat_conf.get("vol_window", 5))
    data_conf = conf.get("data", {}) if isinstance(conf, dict) else {}
    tickers_conf: List[str] = [
        str(t).strip().upper() for t in data_conf.get("tickers", []) if str(t).strip()
    ]
    start_date: Optional[str] = data_conf.get("start_date")
    end_date: Optional[str] = data_conf.get("end_date")

    # -------------------- Price features --------------------
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

    # Price-only panel to merge against
    df_price = df[["date", "ticker"] + feat_cols].sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    # -------------------- Sentiment features --------------------
    def _load_best_sentiment(root_dir: Path) -> Optional[pd.DataFrame]:
        """
        Load available sentiment CSVs and return the one with more (date, ticker) rows.

        Candidates:
        - data/text/sentiment_ai_daily_2023_2025_streamed.csv
        - data/text/sentiment_ai_daily_2024.csv
        """

        candidates: List[Tuple[Path, str]] = [
            (
                root_dir
                / "data"
                / "text"
                / "sentiment_ai_daily_2023_2025_streamed.csv",
                "streamed_2023_2025",
            ),
            (
                root_dir / "data" / "text" / "sentiment_ai_daily_2024.csv",
                "local_2024",
            ),
        ]

        best_df: Optional[pd.DataFrame] = None
        best_rows: int = 0

        for path, _name in candidates:
            if not path.exists():
                continue
            tmp = pd.read_csv(path)
            if "date" not in tmp.columns or "ticker" not in tmp.columns:
                continue

            # Basic cleaning: date to datetime, drop invalids, uppercase tickers
            tmp["date"] = pd.to_datetime(tmp["date"], utc=False, errors="coerce")
            tmp = tmp.dropna(subset=["date"])
            tmp["ticker"] = tmp["ticker"].astype(str).str.upper().str.strip()

            # Optional filter to configured tickers and date range
            if tickers_conf:
                tmp = tmp[tmp["ticker"].isin(tickers_conf)]

            if start_date is not None:
                tmp = tmp[tmp["date"] >= pd.to_datetime(start_date)]
            if end_date is not None:
                tmp = tmp[tmp["date"] <= pd.to_datetime(end_date)]

            # Only keep non-empty (date, ticker) rows
            tmp = tmp.dropna(subset=["ticker"])
            n_rows = len(tmp)
            if n_rows == 0:
                continue

            if n_rows > best_rows:
                best_rows = n_rows
                best_df = tmp

        return best_df

    df_sent = _load_best_sentiment(root)

    # If no sentiment is available, fall back to price-only features
    if df_sent is None or df_sent.empty:
        df_out = df_price
        out_path = root / "features" / "features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(out_path, index=False)
        print(
            f"Wrote {len(df_out):,} price-only feature rows to {out_path} "
            "(no sentiment CSVs found)."
        )
        return

    # Ensure we only keep the union of core sentiment columns we care about
    sentiment_base_cols: List[str] = [
        "senti_pos",
        "senti_neu",
        "senti_neg",
        "senti_score",
        "senti_count",
        "senti_count_z",
        "pct_ai_news",
        "pct_chip_news",
        "pct_reg_news",
        "news_volatility",
    ]
    present_sent_cols: List[str] = [
        c for c in sentiment_base_cols if c in df_sent.columns
    ]
    keep_cols = ["date", "ticker"] + present_sent_cols
    df_sent = df_sent[keep_cols]

    # Enforce unique (date, ticker) by averaging numeric columns if necessary
    if df_sent.duplicated(subset=["ticker", "date"]).any():
        numeric_cols = df_sent.select_dtypes(include=["number"]).columns.tolist()
        group_cols = ["ticker", "date"]
        agg_dict: Dict[str, str] = {c: "mean" for c in numeric_cols if c not in group_cols}
        df_sent = (
            df_sent.groupby(group_cols, observed=True)
            .agg(agg_dict)
            .reset_index()
        )

    # Sort sentiment by ticker/date
    df_sent = df_sent.sort_values(["ticker", "date"]).reset_index(drop=True)

    # -------------------- Rolling z-scores for sentiment --------------------
    # Apply rolling per-ticker z-scores to selected sentiment columns, adding new features.
    sentiment_z_map: Dict[str, str] = {
        "senti_score": "senti_score_z",
        "senti_count": "senti_count_roll_z",
        "news_volatility": "news_volatility_z",
    }
    for src_col, z_col in sentiment_z_map.items():
        if src_col in df_sent.columns:
            df_sent[z_col] = df_sent.groupby("ticker", observed=True)[src_col].transform(
                lambda s: rolling_zscore(s, zscore_window)
            )

    # -------------------- Merge and neutral fill --------------------
    df_merged = df_price.merge(df_sent, on=["date", "ticker"], how="left")

    # Neutral / no-news defaults
    neutral_values: Dict[str, float] = {
        "senti_pos": 1.0 / 3.0,
        "senti_neu": 1.0 / 3.0,
        "senti_neg": 1.0 / 3.0,
        "senti_score": 0.0,
        "senti_count": 0.0,
        "senti_count_z": 0.0,
        "pct_ai_news": 0.0,
        "pct_chip_news": 0.0,
        "pct_reg_news": 0.0,
        "news_volatility": 0.0,
        "senti_score_z": 0.0,
        "senti_count_roll_z": 0.0,
        "news_volatility_z": 0.0,
    }
    for col, val in neutral_values.items():
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(val)

    # Final column ordering: price features first, then sentiment
    sentiment_z_cols: List[str] = [
        "senti_score_z",
        "senti_count_roll_z",
        "news_volatility_z",
    ]
    ordered_sent_cols: List[str] = [
        c for c in sentiment_base_cols + sentiment_z_cols if c in df_merged.columns
    ]
    out_cols = ["date", "ticker"] + feat_cols + ordered_sent_cols
    df_out = df_merged[out_cols].sort_values(["date", "ticker"]).reset_index(drop=True)

    # Basic sanity check: no NaNs in any feature columns
    feature_cols_full = [c for c in out_cols if c not in ("date", "ticker")]
    if df_out[feature_cols_full].isna().any().any():
        raise ValueError("NaNs detected in fused feature table after neutral fill.")

    out_path = root / "features" / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"Wrote {len(df_out):,} fused feature rows to {out_path}")


if __name__ == "__main__":
    main()


