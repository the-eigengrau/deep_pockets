from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_ppo(env, timesteps=500_000):
    venv = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", venv, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model
