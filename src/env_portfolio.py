import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Reinforcement Learning environment for multi-asset portfolio allocation.

    Key design choices:
    - Observations include ALL features for ALL tickers on that day + current portfolio weights.
    - Actions represent *changes in portfolio weights*, clipped by max_w_change.
    - No shorting: weights are clipped to >= 0 and renormalized.
    - Reward = portfolio daily return * reward_scale - transaction_cost.
    - reward_scale=100 makes tiny financial returns (0.001) usable for RL agents.
    - Gymnasium API enforced: reset() returns (obs, info), step() returns (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        tickers,
        transaction_cost=0.001,
        max_w_change=0.10,
        reward_scale=100,
    ):
        super().__init__()

        self.df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)
        self.transaction_cost = transaction_cost
        self.max_w_change = max_w_change
        self.reward_scale = reward_scale

        # All feature columns EXCEPT date/ticker/adj_close
        self.feature_cols = [
            c for c in df.columns
            if c not in ["date", "ticker", "adj_close"]
        ]

        # Pivot features: shape (num_days x (features_per_asset), assets)
        self.feature_panel = (
            df.pivot(index="date", columns="ticker", values=self.feature_cols)
              .sort_index()
        )

        # Pivot returns — VERY IMPORTANT: ret_1d must already be computed
        self.returns_panel = (
            df.pivot(index="date", columns="ticker", values="ret_1d")
              .sort_index() / 100.0
        )  # convert % to decimal

        self.dates = self.returns_panel.index.tolist()
        self.max_step = len(self.dates) - 2  # final usable step

        # Observation space = flatten(features) + weights
        obs_dim = len(self.feature_cols) * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space = delta weights per asset
        self.action_space = spaces.Box(
            low=-self.max_w_change,
            high=self.max_w_change,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Internal state
        self._current_step = None
        self._w = None
        self.last_raw_reward = 0.0

    # ============================
    # Gymnasium reset()
    # ============================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._current_step = 0
        self._w = np.ones(self.n_assets) / self.n_assets  # equal-weight start

        obs = self._get_obs()
        info = {}
        return obs, info

    # ============================
    # Gymnasium step()
    # ============================
    def step(self, action):
        # --- Process action (rebalance weights) ---
        delta_w = np.clip(action, -self.max_w_change, self.max_w_change)
        new_w = self._w + delta_w

        # No shorting
        new_w = np.clip(new_w, 0, 1)

        # Normalize to sum=1 (full investment)
        if new_w.sum() == 0:
            new_w = np.ones(self.n_assets) / self.n_assets
        else:
            new_w = new_w / new_w.sum()

        # --- Transaction cost (L1 turnover) ---
        turnover = np.abs(new_w - self._w).sum()
        tc_cost = self.transaction_cost * turnover

        # --- Portfolio return ---
        raw_ret_vec = self.returns_panel.iloc[self._current_step].values
        raw_reward = float(np.dot(new_w, raw_ret_vec) - tc_cost)

        # Save raw reward for NAV reconstruction
        self.last_raw_reward = raw_reward

        # Scale reward for RL agent stability
        reward = raw_reward * self.reward_scale

        # --- Update state ---
        self._w = new_w
        self._current_step += 1

        # Episode termination
        terminated = self._current_step >= self.max_step
        truncated = False

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    # ============================
    # Helper: Build observation vector
    # ============================
    def _get_obs(self):
        row_feats = self.feature_panel.iloc[self._current_step].values.flatten()
        obs = np.concatenate([row_feats, self._w], dtype=np.float32)
        return obs

    def render(self):
        pass
