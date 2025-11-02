"""
Stub: PPO agent and tiny MLP policy for portfolio weights.

Design (to implement later):
- Policy network: MLP [128, 64] with ReLU and optional dropout 0.1.
- Head: linear logits per ticker → softmax to long-only weights.
- PPO hyperparams per conf.yaml (lr, gamma, gae_lambda, clip, epochs).
"""

from dataclasses import dataclass

import torch  # type: ignore
import torch.nn as nn  # type: ignore


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    epochs: int = 5
    hidden_sizes: tuple[int, int] = (128, 64)
    dropout: float = 0.1


class PolicyMLP(nn.Module):  # Placeholder network signature
    def __init__(self, input_dim: int, num_tickers: int, config: PPOConfig) -> None:
        super().__init__()
        # To be implemented later
        self.input_dim = input_dim
        self.num_tickers = num_tickers
        self.config = config


