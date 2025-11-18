"""
build_news_features_csv.py

Generic pipeline:
  - Load a local CSV of news (from Kaggle, etc.)
  - Columns required: a date column, a text/headline column
  - Filter by date range
  - Map each row to 0+ tickers based on name/cashtag rules
  - Run FinBERT on the headlines
  - Aggregate per (ticker, date) to build daily sentiment features
  - Lag features by 1 day to avoid look-ahead
  - Save sentiment_ai_daily.csv

Output columns:
  ticker, date,
  senti_pos, senti_neu, senti_neg, senti_score,
  senti_count, senti_count_z,
  pct_ai_news, pct_chip_news, pct_reg_news,
  news_volatility
"""

import argparse
import re
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------------------------------------------------
# CONFIG: AI tickers & company names
# ---------------------------------------------------------------------

DEFAULT_TICKERS = ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMD', 'ASML', 'PLTR', 'ORCL', 'SNOW', 'TSM']

COMPANY_KEYWORDS: Dict[str, List[str]] = {
    "NVDA": ["nvidia", "nvda"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "META": ["meta", "facebook", "meta platforms"],
    "AMD": ["amd", "advanced micro devices"],
    "ASML": ["asml"],
    "PLTR": ["palantir", "pltr"],
    "ORCL": ["oracle", "orcl"],
    "SNOW": ["snowflake", "snow"],
    "TSM": ["tsm", "tsmc", "taiwan semiconductor"],
}

AI_PATTERN = re.compile(
    r"\b(ai|artificial intelligence|machine learning|deep learning|chatgpt|llm|generative ai)\b",
    re.IGNORECASE,
)
CHIP_PATTERN = re.compile(
    r"\b(chip|chips|gpu|gpus|accelerator|semiconductor|semiconductors|foundry|wafer)\b",
    re.IGNORECASE,
)
REG_PATTERN = re.compile(
    r"\b(regulation|regulatory|antitrust|lawsuit|ban|bans|export control|sanction|sanctions)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------
# FinBERT wrapper
# ---------------------------------------------------------------------

class FinBertSentiment:
    def __init__(self, model_id: str = "yiyanghkust/finbert-tone", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            use_safetensors=True,
            trust_remote_code=False
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts: List[str], batch_size: int = 32, max_length: int = 64) -> np.ndarray:
        all_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT batches"):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

# ---------------------------------------------------------------------
# Helpers: ticker mapping, keyword flags, aggregation, lag
# ---------------------------------------------------------------------

def build_ticker_patterns(tickers: List[str]) -> Dict[str, re.Pattern]:
    patterns = {}
    for tic in tickers:
        names = COMPANY_KEYWORDS.get(tic, [tic])
        cashtag = rf"\${tic}\b"
        name_part = "|".join(re.escape(n) for n in names)
        pat = re.compile(rf"({cashtag})|({name_part})", re.IGNORECASE)
        patterns[tic] = pat
    return patterns

def map_tickers(text_series: pd.Series, tickers: List[str]) -> pd.Series:
    patterns = build_ticker_patterns(tickers)
    mapped = []
    for txt in text_series.fillna(""):
        hits = [tic for tic, p in patterns.items() if p.search(txt)]
        mapped.append(hits)
    return pd.Series(mapped, index=text_series.index)

def add_keyword_flags(df: pd.DataFrame) -> pd.DataFrame:
    text = df["headline"].fillna("")
    df["ai_flag"] = text.str.contains(AI_PATTERN)
    df["chip_flag"] = text.str.contains(CHIP_PATTERN)
    df["reg_flag"] = text.str.contains(REG_PATTERN)
    return df

def aggregate_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df["seen_datetime"].dt.date
    df["senti_score"] = df["prob_pos"] - df["prob_neg"]

    grouped = (
        df.groupby(["ticker", "date"])
        .agg(
            senti_pos=("prob_pos", "mean"),
            senti_neu=("prob_neu", "mean"),
            senti_neg=("prob_neg", "mean"),
            senti_score=("senti_score", "mean"),
            senti_count=("headline", "count"),
            pct_ai_news=("ai_flag", "mean"),
            pct_chip_news=("chip_flag", "mean"),
            pct_reg_news=("reg_flag", "mean"),
            news_volatility=("senti_score", "std"),
        )
        .reset_index()
    )

    grouped["news_volatility"] = grouped["news_volatility"].fillna(0.0)
    grouped["senti_count"] = grouped["senti_count"].astype(float)
    grouped["date"] = pd.to_datetime(grouped["date"])
    grouped = grouped.sort_values(["ticker", "date"])

    def zscore(series: pd.Series) -> pd.Series:
        m = series.mean()
        s = series.std()
        if s == 0 or np.isnan(s):
            return pd.Series(np.zeros_like(series), index=series.index)
        return (series - m) / s

    grouped["senti_count_z"] = grouped.groupby("ticker")["senti_count"].transform(zscore)
    return grouped

def lag_features(df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"])
    feature_cols = [c for c in df.columns if c not in ["ticker", "date"]]

    df[feature_cols] = df.groupby("ticker")[feature_cols].shift(lag_days)

    neutral_fill = {
        "senti_pos": 1/3,
        "senti_neu": 1/3,
        "senti_neg": 1/3,
        "senti_score": 0.0,
        "senti_count": 0.0,
        "senti_count_z": 0.0,
        "pct_ai_news": 0.0,
        "pct_chip_news": 0.0,
        "pct_reg_news": 0.0,
        "news_volatility": 0.0,
    }
    for c, val in neutral_fill.items():
        if c in df.columns:
            df[c] = df[c].fillna(val)

    return df

# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

def build_news_features_csv(
    input_csv: str,
    date_col: str,
    text_col: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_path: str = "sentiment_ai_daily.csv",
) -> pd.DataFrame:
    print(f"Loading news from {input_csv}...")
    df = pd.read_csv(input_csv)

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in CSV columns: {df.columns.tolist()}")
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV columns: {df.columns.tolist()}")

    df = df[[date_col, text_col]].rename(columns={date_col: "date_raw", text_col: "headline"})

    # parse dates and filter
    df["seen_datetime"] = pd.to_datetime(df["date_raw"], errors="coerce", utc=True)
    df["seen_datetime"] = df["seen_datetime"].dt.tz_localize(None)
    df = df.dropna(subset=["seen_datetime"])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    df = df[(df["seen_datetime"] >= start_dt) & (df["seen_datetime"] < end_dt)]
    print(f"After date filter {start_date} → {end_date}: {len(df)} rows")

    if df.empty:
        raise RuntimeError("No articles in requested date range; check date_col, date format, and range.")

    # map to tickers
    print(f"Mapping articles to {len(tickers)} tickers...")
    df["ticker_list"] = map_tickers(df["headline"], tickers)
    df = df[df["ticker_list"].str.len() > 0]
    df = df.explode("ticker_list").rename(columns={"ticker_list": "ticker"})

    print(f"After ticker mapping: {len(df)} article-ticker pairs")
    if df.empty:
        raise RuntimeError("No articles matched your tickers; check COMPANY_KEYWORDS or text_col.")

    # topic flags
    df = add_keyword_flags(df)

    # FinBERT
    print(f"Running FinBERT on {len(df)} headlines...")
    finbert = FinBertSentiment()
    probs = finbert.predict_proba(df["headline"].tolist())
    df[["prob_neg", "prob_neu", "prob_pos"]] = probs

    # aggregate daily
    print("Aggregating to daily per (ticker, date)...")
    daily = aggregate_daily_features(df)

    # lag
    print("Applying 1-day lag to avoid look-ahead...")
    daily_lagged = lag_features(daily, lag_days=1)

    # clean date
    daily_lagged["date"] = daily_lagged["date"].dt.strftime("%Y-%m-%d")

    print(f"Saving features to {output_path}")
    daily_lagged.to_csv(output_path, index=False)
    print("Done.")
    return daily_lagged

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build daily news sentiment features from a local CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to local news CSV (e.g., Kaggle dataset).")
    parser.add_argument("--date_col", type=str, required=True, help="Name of the date column in the CSV.")
    parser.add_argument("--text_col", type=str, required=True, help="Name of the text/headline column in the CSV.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="*",
        default=DEFAULT_TICKERS,
        help="List of tickers to filter (default: AI basket)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sentiment_ai_daily.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    build_news_features_csv(
        input_csv=args.input_csv,
        date_col=args.date_col,
        text_col=args.text_col,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
    )


'''

python get_headline.py --input_csv financial_news_2024.csv --date_col date --text_col title --start_date 2024-01-01 --end_date   2024-12-31 --output sentiment_ai_daily_2024.csv


'''