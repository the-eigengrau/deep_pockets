import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env_portfolio import PortfolioEnv


# ============================================
# Train/Test split
# ============================================
def split_data(df, train_start, train_end, test_end):
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    test_df  = df[(df["date"] > train_end) & (df["date"] <= test_end)].copy()
    return train_df, test_df


# ============================================
# Helper: PPO linear LR/clip schedule
# ============================================
def linear_schedule(initial_value):
    """
    PPO benefits from decreasing LR and clip_range during training.
    'progress' goes from 1 → 0.
    """
    def func(progress):
        return progress * initial_value
    return func


# ============================================
# Evaluate trained agent
# ============================================
def evaluate(model, env, initial_value=100000):
    """
    Evaluation for VecEnv:
    - reset() returns obs ONLY (VecNormalize API)
    - step() returns obs, reward, done, info
    """
    obs = env.reset()
    done = False
    nav = [initial_value]
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        r = float(reward[0])
        rewards.append(r)

        # info is a list of dicts in DummyVecEnv
        info_dict = info[0] if isinstance(info, list) else info

        # get raw_reward stored by the environment (unscaled daily return)
        raw_ret = info_dict.get("raw_reward", r / 100.0)

        # NAV update uses unscaled returns
        nav.append(nav[-1] * (1 + raw_ret))

    return np.array(nav), np.array(rewards)


# ============================================
# Compute Sharpe / MDD / CumReturn
# ============================================
def compute_metrics(nav):
    ret = nav[1:] / nav[:-1] - 1
    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252) if np.std(ret) > 0 else 0

    running_max = np.maximum.accumulate(nav)
    mdd = (nav / running_max - 1).min()

    cum = nav[-1] / nav[0] - 1
    return {"cumulative": cum, "sharpe": sharpe, "mdd": mdd}


# ============================================
# Main experiment runner
# ============================================
def run_experiment(agent_name, df, train_start, train_end, test_end,
                   timesteps=300_000, tc=0.001, max_w_change=0.10,
                   reward_scale=100):

    tickers = df["ticker"].unique()
    train_df, test_df = split_data(df, train_start, train_end, test_end)

    # ============================
    # Build training environment
    # ============================
    train_env = DummyVecEnv([
        lambda: PortfolioEnv(
            train_df,
            tickers,
            transaction_cost=tc,
            max_w_change=max_w_change,
            reward_scale=reward_scale
        )
    ])

    # VecNormalize = observation + reward normalization
    train_env = VecNormalize(train_env,
                             norm_obs=True,
                             norm_reward=True,
                             clip_obs=10.)
    
    # ============================
    # PPO uses CPU; SAC/TD3 use GPU
    # ============================
    if agent_name.lower() == "ppo":
        device = "cpu"
        algo_class = PPO
        model = algo_class(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=device,
            learning_rate=linear_schedule(3e-4),
            clip_range=linear_schedule(0.2),
        )
    elif agent_name.lower() == "sac":
        device = "cuda"
        algo_class = SAC
        model = algo_class(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            batch_size=256,
            tau=0.005,
        )
    elif agent_name.lower() == "td3":
        device = "cuda"
        algo_class = TD3
        model = algo_class(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            policy_delay=2,
            batch_size=256,
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # ============================
    # Train agent
    # ============================
    model.learn(total_timesteps=timesteps)

    # Save normalizer so test_env matches training normalization
    train_env.save("normalizer.pkl")

    # ============================
    # Build test environment
    # ============================
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
    test_env.norm_reward = False  # raw returns on test

    # ============================
    # Evaluate
    # ============================
    train_nav, train_rewards = evaluate(model, train_env)
    test_nav,  test_rewards  = evaluate(model, test_env)

    # ============================
    # Save results
    # ============================
    out_dir = f"./figs/rl_results/{agent_name}/"
    os.makedirs(out_dir, exist_ok=True)

    # NAV curve
    plt.figure(figsize=(14, 6))
    plt.plot(train_nav, label="Train NAV")
    plt.plot(range(len(train_nav), len(train_nav) + len(test_nav)), test_nav, label="Test NAV")
    plt.title(f"{agent_name.upper()} NAV Curve")
    plt.legend()
    plt.savefig(out_dir + "nav_curve.png")
    plt.close()

    # Reward curve
    plt.figure(figsize=(14, 6))
    plt.plot(train_rewards, label="Train Reward")
    plt.plot(test_rewards, label="Test Reward")
    plt.title(f"{agent_name.upper()} Reward Curve")
    plt.legend()
    plt.savefig(out_dir + "reward_curve.png")
    plt.close()

    metrics = pd.DataFrame({
        "train": compute_metrics(train_nav),
        "test": compute_metrics(test_nav)
    })
    metrics.to_csv(out_dir + "metrics.csv")

    return metrics


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "sac", "td3"])
    parser.add_argument("--data", type=str, default="./data/features_with_prices.csv")

    parser.add_argument("--train_start", type=str, default="2023-01-01")
    parser.add_argument("--train_end", type=str, default="2024-06-30")
    parser.add_argument("--test_end", type=str, default="2025-11-01")

    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--tc", type=float, default=0.001)
    parser.add_argument("--max_w_change", type=float, default=0.10)
    parser.add_argument("--reward_scale", type=float, default=100)

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])

    results = run_experiment(
        agent_name=args.agent,
        df=df,
        train_start=args.train_start,
        train_end=args.train_end,
        test_end=args.test_end,
        timesteps=args.timesteps,
        tc=args.tc,
        max_w_change=args.max_w_change,
        reward_scale=args.reward_scale
    )

    print("\n=== FINAL RESULTS ===")
    print(results)
