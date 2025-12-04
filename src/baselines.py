import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "./data/cleaned_features_23-25.csv"
FIG_PATH = "./figs"
os.makedirs(FIG_PATH, exist_ok=True)

# ============================
# Utility Functions
# ============================

def compute_nav(returns, start_val=100000):
    """Convert series of daily returns to NAV."""
    return start_val * (1 + returns).cumprod()


def compute_dd(nav):
    """Compute drawdown timeseries."""
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return dd


def summary_stats(returns):
    returns = returns.dropna()
    ann_ret = (1 + returns.mean())**252 - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    nav = (1 + returns).cumprod()
    mdd = compute_dd(nav).min()
    cum = nav.iloc[-1] - 1
    return pd.Series([cum, ann_ret, vol, sharpe, mdd],
                     index=["Cumulative", "Annualized", "Volatility", "Sharpe", "MDD"])


# ============================
# Load Data
# ============================

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

tickers = df["ticker"].unique()
print("Tickers used:", tickers)

# Pivot for returns
price_pivot = df.pivot(index="date", columns="ticker", values="adj_close")
ret_1d = price_pivot.pct_change().fillna(0)


# ============================
# Strategy Implementations
# ============================

def equal_weight_strategy(freq="D", tc=0.0):
    """
    freq: 'D' = daily, 'W' = weekly, 'M' = monthly
    tc: transaction costs (e.g. 0.0005 = 5 bps)
    """
    dates = price_pivot.index
    weights = pd.DataFrame(0, index=dates, columns=tickers)

    current_w = np.ones(len(tickers)) / len(tickers)

    last_reb = None

    for d in dates:
        if last_reb is None or d.to_period(freq) != last_reb.to_period(freq):
            # rebalance
            new_w = np.ones(len(tickers)) / len(tickers)

            # transaction costs
            turnover = np.abs(new_w - current_w).sum()
            cost = turnover * tc

            # store weights
            weights.loc[d] = new_w
            current_w = new_w

            last_reb = d
        else:
            weights.loc[d] = current_w

    # compute portfolio returns
    port_ret = (weights.shift().fillna(weights.iloc[0]) * ret_1d).sum(axis=1)
    port_ret -= tc / len(dates)  # distribute cost (small)

    return port_ret


def momentum_strategy(tc=0.0):
    # rank by mom20 each day
    mom20 = df.pivot(index="date", columns="ticker", values="mom20")

    weights = (mom20.rank(axis=1, pct=True))
    # convert rank to allocation (0 to 1)
    weights = weights.div(weights.sum(axis=1), axis=0)

    # simple transaction cost approximation
    turnover = weights.diff().abs().sum(axis=1)
    port_ret = (weights.shift().fillna(weights.iloc[0]) * ret_1d).sum(axis=1)
    port_ret -= tc * turnover

    return port_ret


def sentiment_strategy(tc=0.0):
    sent = df.pivot(index="date", columns="ticker", values="senti_score_z").fillna(0)

    # convert sentiment signals to weights
    raw = np.maximum(sent, 0)
    raw = raw.replace(0, np.nan).fillna(0)
    weights = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    turnover = weights.diff().abs().sum(axis=1)
    port_ret = (weights.shift().fillna(weights.iloc[0]) * ret_1d).sum(axis=1)
    port_ret -= tc * turnover
    return port_ret


def random_strategy(tc=0.0, seed=42):
    np.random.seed(seed)
    dates = price_pivot.index
    weights = pd.DataFrame(index=dates, columns=tickers)

    prev_w = None

    for d in dates:
        w = np.random.random(len(tickers))
        w = w / w.sum()
        weights.loc[d] = w

        if prev_w is None:
            prev_w = w
            continue

        turnover = np.abs(w - prev_w).sum()
        prev_w = w

    port_ret = (weights.shift().fillna(weights.iloc[0]) * ret_1d).sum(axis=1)
    port_ret -= tc / len(dates)
    return port_ret


# ============================
# RUN ALL BASELINES
# ============================

strategies = {
    "EW Daily": equal_weight_strategy(freq="D", tc=0.0005),
    "EW Monthly": equal_weight_strategy(freq="M", tc=0.0005),
    "Momentum": momentum_strategy(tc=0.0005),
    "Sentiment": sentiment_strategy(tc=0.0005),
    "Random": random_strategy(tc=0.0005),
}

navs = pd.DataFrame({k: compute_nav(v) for k, v in strategies.items()})
rets = pd.DataFrame(strategies)

# ============================
# SAVE Summary Table
# ============================

summary = rets.apply(summary_stats)
summary.to_csv(f"{FIG_PATH}/baseline_summary.csv")
print(summary)

# ============================
# PLOTS
# ============================

# NAV CURVES
plt.figure(figsize=(14,6))
for k in navs.columns:
    plt.plot(navs.index, navs[k], label=k)
plt.title("Portfolio NAV Curves")
plt.ylabel("NAV ($)")
plt.legend()
plt.savefig(f"{FIG_PATH}/nav_curves.png")
plt.close()

# DRAWDOWNS
plt.figure(figsize=(14,6))
for k in navs.columns:
    dd = compute_dd(navs[k])
    plt.plot(dd, label=k)
plt.title("Drawdown Curves")
plt.legend()
plt.savefig(f"{FIG_PATH}/drawdowns.png")
plt.close()

# ROLLING SHARPE
plt.figure(figsize=(14,6))
for k in rets.columns:
    rs = rets[k].rolling(63).mean() / rets[k].rolling(63).std()
    plt.plot(rs, label=k)
plt.title("Rolling Sharpe (3M)")
plt.legend()
plt.savefig(f"{FIG_PATH}/rolling_sharpe.png")
plt.close()

# CORRELATION HEATMAP
plt.figure(figsize=(8,6))
sns.heatmap(rets.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation of Baseline Strategies")
plt.savefig(f"{FIG_PATH}/baseline_corr.png")
plt.close()





"""
baselines.py
Generates baseline strategies, NAV curves, drawdowns, correlation heatmaps,
and summary statistics for the project.

Reads:
    ./data/features_with_prices.csv

Outputs:
    ./figs/*.png
    ./figs/baseline_summary_table.csv
"""
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# PATHS
# ============================================================
DATA_PATH = "./data/features_with_prices.csv"
FIG_DIR = "./figs/"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading merged dataset with prices...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["date", "ticker"])

# Filter to project period
df = df[df["date"] >= "2023-01-01"]

tickers = df["ticker"].unique().tolist()
print("Tickers used:", tickers)

# ============================================================
# 2. PRICE + RETURN MATRICES
# ============================================================

# Pivot adj_close into price matrix
price_pivot = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()

# Daily returns (decimal)
ret = price_pivot.pct_change().fillna(0)

# ============================================================
# 3. NAV ENGINE
# ============================================================

def compute_nav(weights, ret_df, initial_cap=100000):
    """
    weights: DataFrame (index=date, columns=tickers) row-normalized
    """
    weights = weights.reindex(ret_df.index).fillna(0)
    weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0)

    daily_ret = (weights * ret_df).sum(axis=1)
    nav = initial_cap * (1 + daily_ret).cumprod()

    return nav, daily_ret


# ============================================================
# 4. BASELINE STRATEGIES
# ============================================================

print("Computing baseline strategies...")

# ---- Equal Weight DAILY ----
ew_daily_w = pd.DataFrame(1/len(tickers), index=ret.index, columns=tickers)
ew_daily_nav, ew_daily_ret = compute_nav(ew_daily_w, ret)

# ---- Equal Weight MONTHLY ----
ew_monthly_w = pd.DataFrame(0, index=ret.index, columns=tickers)
prev_month = None
current_w = np.ones(len(tickers)) / len(tickers)

for dt in ret.index:
    if prev_month != dt.month:
        current_w = np.ones(len(tickers)) / len(tickers)
    ew_monthly_w.loc[dt] = current_w
    prev_month = dt.month

ew_monthly_nav, ew_monthly_ret = compute_nav(ew_monthly_w, ret)

# ---- Momentum (20-day) ----
mom_signal = df.pivot(index="date", columns="ticker", values="mom20").sort_index()
mom_rank = mom_signal.rank(axis=1, pct=True)
mom_w = mom_rank.div(mom_rank.sum(axis=1), axis=0).fillna(0)
mom_nav, mom_ret = compute_nav(mom_w, ret)

# ---- Sentiment Z-score weighted ----
sent_signal = df.pivot(index="date", columns="ticker", values="senti_score_z").sort_index().fillna(0)
sent_w = sent_signal.div(sent_signal.abs().sum(axis=1), axis=0).fillna(0)
sent_nav, sent_ret = compute_nav(sent_w, ret)

# ---- Random Strategy ----
np.random.seed(42)
rand_w = pd.DataFrame(np.random.randn(*ret.shape), index=ret.index, columns=tickers)
rand_w = rand_w.div(rand_w.abs().sum(axis=1), axis=0)
rand_nav, rand_ret = compute_nav(rand_w, ret)

# ---- Buy & Hold per ticker (price-based) ----
buyhold_nav = {}
for t in tickers:
    p = price_pivot[t]
    nav = 100000 * p / p.iloc[0]     # correct way to do buy & hold
    buyhold_nav[t] = nav


# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================

def summary(returns):
    r = returns.dropna()
    cum = (1 + r).prod() - 1
    ann = (1 + cum)**(252/len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 1e-8 else 0
    wealth = (1 + r).cumprod()
    max_dd = ((wealth - wealth.cummax()) / wealth.cummax()).min()
    return [cum, ann, vol, sharpe, max_dd]

summary_df = pd.DataFrame({
    "EW Daily": summary(ew_daily_ret),
    "EW Monthly": summary(ew_monthly_ret),
    "Momentum": summary(mom_ret),
    "Sentiment": summary(sent_ret),
    "Random": summary(rand_ret),
}, index=["Cumulative Return", "Annualized Return", "Volatility", "Sharpe", "Max Drawdown"])

summary_df.to_csv(os.path.join(FIG_DIR, "baseline_summary_table.csv"))
print("Saved baseline stats → figs/baseline_summary_table.csv")


# ============================================================
# 6. FIGURES
# ============================================================

print("Generating figures...")

# ---- Portfolio NAV curves ----
plt.figure(figsize=(14,6))
plt.plot(ew_daily_nav, label="EW Daily")
plt.plot(ew_monthly_nav, label="EW Monthly")
plt.plot(mom_nav, label="Momentum")
plt.plot(sent_nav, label="Sentiment")
plt.plot(rand_nav, label="Random")
plt.title("Portfolio NAV Curves")
plt.ylabel("NAV ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "nav_all.png"))
plt.close()

# ---- Buy & Hold NAV ----
plt.figure(figsize=(14,6))
for t in tickers:
    plt.plot(buyhold_nav[t], label=t)
plt.title("Buy & Hold NAV by Ticker")
plt.ylabel("NAV ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "buyhold.png"))
plt.close()

# ---- Rolling Sharpe ----
window = 63
plt.figure(figsize=(14,6))
plt.plot(ew_daily_ret.rolling(window).mean() / ew_daily_ret.rolling(window).std(), label="EW Daily")
plt.plot(ew_monthly_ret.rolling(window).mean() / ew_monthly_ret.rolling(window).std(), label="EW Monthly")
plt.plot(mom_ret.rolling(window).mean() / mom_ret.rolling(window).std(), label="Momentum")
plt.plot(sent_ret.rolling(window).mean() / sent_ret.rolling(window).std(), label="Sentiment")
plt.title("Rolling Sharpe (3M)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rolling_sharpe.png"))
plt.close()

# ---- Drawdown ----
def drawdown(nav):
    peak = nav.cummax()
    return (nav - peak) / peak

plt.figure(figsize=(14,6))
plt.plot(drawdown(ew_daily_nav), label="EW Daily")
plt.plot(drawdown(ew_monthly_nav), label="EW Monthly")
plt.plot(drawdown(mom_nav), label="Momentum")
plt.plot(drawdown(sent_nav), label="Sentiment")
plt.plot(drawdown(rand_nav), label="Random")
plt.title("Drawdown Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "drawdown.png"))
plt.close()

# ---- Correlation between baseline returns ----
baseline_rets = pd.DataFrame({
    "EW Daily": ew_daily_ret,
    "EW Monthly": ew_monthly_ret,
    "Momentum": mom_ret,
    "Sentiment": sent_ret,
    "Random": rand_ret,
})

plt.figure(figsize=(8,6))
sns.heatmap(baseline_rets.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation of Baseline Strategies")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "baseline_corr.png"))
plt.close()

print("All baseline figures saved to ./figs/")
'''