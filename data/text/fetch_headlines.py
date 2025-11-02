"""
Stub: Ingest headlines/tweets and map to (date, ticker, text) rows.

Intended behavior (to implement later):
- Read external sources (e.g., Kaggle datasets) with timestamps and tickers.
- Normalize to schema: [date (YYYY-MM-DD), ticker, text].
- Save consolidated parquet at data/text/headlines_raw.parquet.

Notes:
- Do not scrape in this script; assume local CSV/JSON inputs.
- Deduplicate near-identical texts; drop empty lines.
"""

from pathlib import Path
from typing import List

import pandas as pd  # type: ignore


def main() -> None:
    """Placeholder entrypoint for Adi's headline ingestion."""
    pass


if __name__ == "__main__":
    main()


