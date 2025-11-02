"""
Stub: Baseline portfolio policies.

Planned baselines (to implement later):
- Equal Weight (EW): uniform weights across tickers.
- Momentum-only: softmax of mom20 per day.
- Sentiment-only: softmax of (senti_pos - senti_neg).
- Teacher (optional): linear regression on features → softmax weights.
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


