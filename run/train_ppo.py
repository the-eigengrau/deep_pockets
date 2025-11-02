"""
Stub: Training entrypoint for PPO agent.

Intended behavior (to implement later):
- Load config from conf.yaml.
- Prepare TradingEnv with walk-forward splits.
- Initialize PPO agent and train with early stopping on validation Sharpe.
- Save best checkpoint and training logs.
"""

from pathlib import Path

import yaml  # type: ignore


def main() -> None:
    """Placeholder training script."""
    pass


if __name__ == "__main__":
    main()


