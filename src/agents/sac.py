from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

def train_sac(env, timesteps=500_000):
    venv = DummyVecEnv([lambda: env])
    model = SAC("MlpPolicy", venv, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model
