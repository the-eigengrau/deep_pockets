# deep_pockets
Deep Pockets — Deep RL for AI Stock Portfolio Optimization
=========================================================

Repo scaffold for a 4-week sprint. Files are stubs with comments and minimal imports only.

Environment
-----------
1) Create conda env

   conda env create -f environment.yml
   conda activate deep-pockets

2) Install Python deps using uv inside the conda env

   uv pip install -r requirements.txt

Week 1 (Data & Features)
------------------------
- Ayon: populate prices via `data/prices/fetch_prices.py` → `data/prices/prices.parquet`.
- Adi: populate headlines via `data/text/fetch_headlines.py` → `data/text/headlines_raw.parquet`.
- Build features via `features/build_features.py` → `features/features.parquet`.

Data lag rules
--------------
- Price features use rolling windows and are z-scored per ticker without look-ahead.
- Text features (FinBERT) are pooled by day/ticker and lagged by 1 day before use.

Config
------
- See `conf.yaml` for universe, dates, costs, and PPO defaults.
