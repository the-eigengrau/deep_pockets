import os
import argparse
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.env_portfolio import PortfolioEnv


# ============================
# Helper: Data split
# ============================
def split_data(df, train_start, train_end, test_end):
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    test_df  = df[(df["date"] > train_end) & (df["date"] <= test_end)].copy()
    return train_df, test_df


# ============================
# Helper: PPO linear LR & clip schedules
# ============================
def linear_schedule(initial_value):
    def func(progress):
        return progress * initial_value
    return func


# ============================
# Evaluation
# ============================
def evaluate(model, env, initial_value=100000):
    obs = env.reset()
    done = False
    nav = [initial_value]
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        r = float(reward[0])
        rewards.append(r)

        raw_ret = info.get("raw_reward", r / 100.0)
        nav.append(nav[-1] * (1 + raw_ret))

    return np.array(nav), np.array(rewards)


# ============================
# Metrics
# ============================
def compute_metrics(nav):
    ret = nav[1:] / nav[:-1] - 1
    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252) if np.std(ret) > 0 else 0
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    cum = nav[-1] / nav[0] - 1
    return {"cumulative": cum, "sharpe": sharpe, "mdd": mdd}


# ============================
# Train/Eval one configuration
# ============================
def run_one(agent_name, df, tickers, train_start, train_end, test_end,
            lr, batch_size, gamma, tau, reward_scale=100, tc=0.001,
            max_w_change=0.10, timesteps=200000):

    # Split data
    train_df, test_df = split_data(df, train_start, train_end, test_end)

    # Train env
    train_env = DummyVecEnv([
        lambda: PortfolioEnv(
            train_df,
            tickers,
            transaction_cost=tc,
            max_w_change=max_w_change,
            reward_scale=reward_scale
        )
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Agent selection
    if agent_name == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(lr),
            clip_range=linear_schedule(0.2),
            batch_size=batch_size,
            gamma=gamma,
            verbose=0,
            device="cpu",
        )
    elif agent_name == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            verbose=0,
            device="cuda",
        )
    elif agent_name == "td3":
        model = TD3(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            verbose=0,
            device="cuda",
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # Train
    model.learn(total_timesteps=timesteps)

    # Save normalizer for test
    train_env.save("normalizer.pkl")

    # Test env
    test_env = DummyVecEnv([
        lambda: PortfolioEnv(
            test_df,
            tickers,
            transaction_cost=tc,
            max_w_change=max_w_change,
            reward_scale=reward_scale
        )
    ])
    test_env = VecNormalize.load("normalizer.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False

    # Evaluate
    train_nav, train_rewards = evaluate(model, train_env)
    test_nav,  test_rewards  = evaluate(model, test_env)

    train_metrics = compute_metrics(train_nav)
    test_metrics  = compute_metrics(test_nav)

    return train_metrics, test_metrics, train_nav, test_nav


# ============================
# Sweep Runner
# ============================
def run_sweep(agent_name, df, train_start, train_end, test_end, timesteps):

    tickers = df["ticker"].unique()
    results = []

    # Define sweep grid
    lrs = [1e-4, 3e-4]
    batches = [128, 256]
    gammas = [0.95, 0.99]
    taus = [0.01, 0.005] if agent_name in ["sac", "td3"] else [None]

    sweep = itertools.product(lrs, batches, gammas, taus)

    out_dir = f"./figs/rl_sweep/{agent_name}/"
    os.makedirs(out_dir, exist_ok=True)

    for lr, batch, gamma, tau in sweep:
        tau_str = tau if tau is not None else "NA"

        print(f"\n=== Running {agent_name.upper()} | lr={lr} | batch={batch} | gamma={gamma} | tau={tau}")

        train_m, test_m, train_nav, test_nav = run_one(
            agent_name,
            df,
            tickers,
            train_start,
            train_end,
            test_end,
            lr,
            batch,
            gamma,
            tau,
            timesteps=timesteps,
        )

        # Save plots
        plt.figure(figsize=(14, 6))
        plt.plot(train_nav, label="Train")
        plt.plot(range(len(train_nav), len(train_nav)+len(test_nav)), test_nav, label="Test")
        plt.title(f"{agent_name.upper()} NAV | lr={lr} | batch={batch} | gamma={gamma} | tau={tau_str}")
        plt.legend()
        plt.savefig(out_dir + f"nav_lr{lr}_b{batch}_g{gamma}_t{tau_str}.png")
        plt.close()

        row = {
            "agent": agent_name,
            "lr": lr,
            "batch": batch,
            "gamma": gamma,
            "tau": tau_str,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }
        results.append(row)

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir + "sweep_results.csv", index=False)
    print("Saved sweep results →", out_dir + "sweep_results.csv")


# ============================
# Main CLI
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["ppo", "sac", "td3"], required=True)
    parser.add_argument("--data", type=str, default="./data/cleaned_features_updated_23-25.csv")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--train_start", type=str, default="2023-01-01")
    parser.add_argument("--train_end", type=str, default="2024-06-30")
    parser.add_argument("--test_end", type=str, default="2025-11-01")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])

    run_sweep(
        args.agent,
        df,
        args.train_start,
        args.train_end,
        args.test_end,
        args.timesteps
    )
