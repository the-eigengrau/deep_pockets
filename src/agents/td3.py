from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def train_td3(env, timesteps=500_000):
    venv = DummyVecEnv([lambda: env])

    noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                              sigma=0.05 * np.ones(env.action_space.shape))
    model = TD3("MlpPolicy", venv, action_noise=noise, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model
