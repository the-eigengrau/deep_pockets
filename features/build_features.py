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

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
import yaml  # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore


def main() -> None:
    """Placeholder entrypoint for feature construction."""
    pass


if __name__ == "__main__":
    main()


