"""
Stub: Gym-style daily trading environment.

Intended design (to implement later):
- Observation: concatenated per-ticker feature vectors (price + sentiment).
- Action: softmax over tickers → long-only target weights that sum to 1.
- Transition: advance one trading day; apply action at close.
- Reward: next-day portfolio return minus costs (5 bps one-way) and turnover penalty.
- Costs: proportional to daily turnover (L1 change in weights) times bps.

Notes:
- Use gymnasium/gym interface for step/reset/spaces.
- Enforce no look-ahead by lagging all features appropriately.
"""

from typing import Any, Dict, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # gym fallback
    import gym  # type: ignore
    from gym import spaces  # type: ignore


class TradingEnv:  # Placeholder class signature; implement gym.Env later
    """Placeholder trading environment to be completed in Week 2."""

    def __init__(self) -> None:
        pass


