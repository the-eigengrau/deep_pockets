# Deep RL for AI Stock Portfolio Optimization

## 1. Introduction
AI stocks often react quickly to news and exhibit strong short term trends. Managing a portfolio over these companies therefore requires both awareness of news and respect for recent price movement. This paper studies a simple question: can a small, interpretable set of price and text signals, used within a reinforcement learning (RL) framework, improve daily portfolio allocation over standard baselines once realistic trading costs are included?

We focus on a fixed set of key AI stocks: NVDA, MSFT, AMD, AVGO, SMCI, GOOGL, META, AAPL, ASML, and TSM. Each trading day, the system chooses portfolio weights that are held for one day. Inputs are price trends (5 day and 20 day momentum) plus 5 day volatility and same day headline sentiment per ticker using FinBERT. Text is lagged by one day to avoid look‑ahead.

Our approach is intentionally compact. We build a custom gym style environment with flat slippage and an explicit turnover penalty. A small PPO agent maps the fused state to long only weights via a softmax head. We evaluate with strict walk forward splits and compare against clear baselines.

## 2. Contributions
- A minimal, transparent feature set that fuses price momentum/volatility with per‑ticker headline sentiment, with correct data lags.
- A simple daily trading environment with realistic frictions (5 bps one way costs) and explicit turnover control.
- A compact PPO policy (two layer MLP) that produces long only weights via softmax, suitable for a small data set.
- A walk forward evaluation protocol with strong baselines: equal weight, momentum only, and sentiment only.

## 3. Method overview (summary)
- State: per ticker features concatenated across the universe: [mom5, mom20, vol5, senti_pos, senti_neu, senti_neg, senti_count]. Price features are z scored per ticker using rolling windows to prevent leakage. Sentiment is averaged per day and lagged by one day.
- Action: a vector of portfolio weights obtained by softmax over policy logits (long only, sum to one).
- Reward: next day portfolio return minus 5 bps trading costs and a turnover penalty.
- Learning: PPO with a small MLP.

## 4. Data
- Stocks: NVDA, MSFT, AMD, AVGO, SMCI, GOOGL, META, AAPL, ASML, TSM.
- Prices: daily OHLCV (2018–2025). Features per ticker/day: 1‑day return, 5‑day momentum, 20‑day momentum, 5‑day volatility; standardized per ticker with rolling windows.
- Text: headlines/tweets/earnings call snippets mapped to tickers and dates. FinBERT provides positive/neutral/negative probabilities; we average by day and keep a count. Sentiment is lagged by one day before use.

## 5. Methods Summary
- Splits: Train 2018–2022, Validation 2023, Test 2024–2025 YTD.
- Baselines: equal weight, momentum only (softmax of 20‑day momentum), sentiment only (softmax of positive minus negative).
- Metrics: CAGR, Sharpe (252‑day), Max Drawdown, Annual Turnover, and trades per day.
- Sanity checks: average weight per name, exposure drift, and cost as a percent of gross PnL.

## 6. Relation to prior work
Classic portfolio theory (Markowitz) and universal portfolios provide strong baselines and useful structure. Recent work applies deep RL often PPO to end‑to‑end portfolio allocation and is available in libraries such as FinRL. On text, finance tuned transformers like FinBERT offer reliable sentiment estimates that add signal beyond technicals, especially at short horizons. Best practice emphasizes walk‑forward testing, explicit costs, and turnover control.