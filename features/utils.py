"""
Stub utilities for feature engineering.

Intended helpers (to implement later):
- compute_returns(df): 1-day returns per ticker.
- rolling_momentum(df, window): price momentum over N days.
- rolling_volatility(df, window): std of 1-day returns over N days.
- rolling_zscore(series, window): per-ticker rolling z-score (no leakage).

All rolling features should avoid look-ahead by aligning with prior windows only.
"""

from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


