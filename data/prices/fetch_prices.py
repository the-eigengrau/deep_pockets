"""
Download daily OHLCV and Adj Close for tickers and date range from conf.yaml
and save to data/prices/prices.parquet.

Output schema (parquet):
[date, ticker, open, high, low, close, adj_close, volume]

Notes:
- Raw data only; no feature engineering here.
- Dates are timezone-naive and normalized to midnight.
"""

from pathlib import Path
import time
import random
from time import sleep
from typing import Dict, Iterable, List
import io

import requests  # type: ignore
from requests.adapters import HTTPAdapter  # type: ignore
from urllib3.util.retry import Retry  # type: ignore

import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore
import yaml  # type: ignore


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(conf_path: Path) -> Dict:
    with conf_path.open("r") as f:
        return yaml.safe_load(f)


def normalize_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df[date_col] = df[date_col].dt.tz_localize(None)
    df[date_col] = df[date_col].dt.normalize()
    return df


def download_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """
    Stooq CSV fetch (no API keys, no cookies). Emits:
    [date, ticker, open, high, low, close, adj_close, volume]
    Notes:
      - Stooq returns daily EOD OHLCV; we set adj_close = close.
      - Tries both 'TICKER' and 'TICKER.US' (Stooq sometimes uses .US).
    """
    cols_out = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame(columns=cols_out)

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    def fetch_one(tic: str) -> pd.DataFrame:
        base = "https://stooq.com/q/d/l/"
        # Try plain and .US variant (e.g., META vs META.US)
        symbols = [tic.lower(), f"{tic.lower()}.us"]
        for sym in symbols:
            try:
                resp = session.get(base, params={"s": sym, "i": "d"}, headers=headers, timeout=30)
                if resp.status_code != 200 or not resp.text or resp.text.startswith("<!"):
                    continue
                raw = pd.read_csv(io.StringIO(resp.text))
                # Expect columns: Date, Open, High, Low, Close, Volume
                if not {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
                    continue

                df = raw.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }).copy()

                # Stooq sometimes returns full history; clip to requested window
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
                df = df.loc[mask]

                # Type clean + schema
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["adj_close"] = df["close"]
                df["ticker"] = tic

                df = df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
                df = df.dropna(subset=["close"])
                if not df.empty:
                    return df
            except Exception:
                # Try next variant
                continue
        # Nothing worked for this ticker
        print(f"Stooq had no data for {tic}")
        return pd.DataFrame(columns=cols_out)

    frames: List[pd.DataFrame] = []
    for tic in tickers:
        frames.append(fetch_one(tic))
        # Tiny pause to be polite to Stooq
        sleep(0.3)

    if not frames:
        return pd.DataFrame(columns=cols_out)

    out = pd.concat(frames, ignore_index=True)
    out = normalize_dates(out, "date")
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Ensure exact columns
    for c in cols_out:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols_out]




def main() -> None:
    root = project_root()
    conf = load_config(root / "conf.yaml")

    tickers = conf.get("data", {}).get("tickers", [])
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]

    start_date = conf.get("data", {}).get("start_date", "2018-01-01")
    end_date = conf.get("data", {}).get("end_date", "2025-11-01")

    print(f"Downloading daily prices for {len(tickers)} tickers from {start_date} to {end_date}...")
    df = download_prices(tickers, start_date, end_date)

    out_path = root / "data" / "prices" / "prices.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
