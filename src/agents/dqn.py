import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=5):
        super().__init__(env)
        self.bins = bins
        self.action_space = gym.spaces.Discrete(env.num_tickers * bins)

    def action(self, a):
        ticker = a // self.bins
        bin_i = a % self.bins
        delta = np.linspace(-self.env.max_w_change, self.env.max_w_change, self.bins)[bin_i]
        action = np.zeros(self.env.num_tickers)
        action[ticker] = delta
        return action

def train_dqn(env, timesteps=500_000):
    env = DiscreteActionWrapper(env)
    venv = DummyVecEnv([lambda: env])
    model = DQN("MlpPolicy", venv, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model
