import argparse
import json
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TICKERS = [
    "NVDA","MSFT","GOOGL","META","AMD","ASML",
    "PLTR","ORCL","SNOW","TSM","AVGO","SMCI","AAPL"
]

AI_WORDS   = ["ai","artificial intelligence","machine learning","deep learning","genai","llm","chatgpt","copilot"]
CHIP_WORDS = ["chip","semiconductor","gpu","accelerator","npu","h100","h200","gh200","foundry","3nm","5nm","7nm"]
REG_WORDS  = ["regulation","ban","doj","ftc","eu probe","sanction","antitrust","lawsuit","export control"]


# ----------------- FINBERT -----------------

class FinBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone",
            use_safetensors=True,
            trust_remote_code=False
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    @torch.no_grad()
    def proba(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tokenizer(
                texts[i:i+bs],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            enc = {k:v.to(self.device) for k,v in enc.items()}
            probs = torch.softmax(self.model(**enc).logits, dim=1)
            out.append(probs.cpu().numpy())
        return np.vstack(out)


# ----------------- STREAMING NEWS -----------------

def extract_headline(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.split("\n\n",1)[0].strip()


def stream_filtered_articles(start_date, end_date):
    """Yields (ticker, date, headline) only for matching rows."""
    start_dt = datetime.fromisoformat(start_date)
    end_dt   = datetime.fromisoformat(end_date)

    ds = load_dataset(
        "Brianferrell787/financial-news-multisource",
        split="train",
        streaming=True
    )

    for row in ds:
        # 1. Parse top-level ISO date
        try:
            dt = datetime.fromisoformat(row["date"].replace("Z","+00:00"))
            dt = dt.replace(tzinfo=None)       # 🚨 FIX: remove timezone
        except:
            continue

        if not (start_dt <= dt <= end_dt):
            continue

        # 2. Parse extra_fields JSON
        try:
            meta = json.loads(row["extra_fields"])
        except:
            continue

        stocks = meta.get("stocks") or meta.get("tickers")
        if not isinstance(stocks, list):
            continue

        # Uppercase symbol matching
        matched = set(s.upper() for s in stocks) & set(TICKERS)
        if not matched:
            continue

        head = extract_headline(row["text"])
        if not head:
            continue

        ds = dt.date().isoformat()

        for tic in matched:
            yield tic, ds, head


# ----------------- BUILD DAILY FEATURES -----------------

def build_daily_features(records):
    """records = list of dicts: ticker,date,title,ai_flag,chip_flag,..."""
    df = pd.DataFrame(records)

    fb = FinBert()
    probs = fb.proba(df["title"].tolist())
    df["neg"], df["neu"], df["pos"] = probs[:,0], probs[:,1], probs[:,2]
    df["score"] = df["pos"] - df["neg"]

    g = df.groupby(["ticker","date"])

    out = g.agg(
        senti_pos=("pos","mean"),
        senti_neu=("neu","mean"),
        senti_neg=("neg","mean"),
        senti_score=("score","mean"),
        senti_count=("title","count"),
        pct_ai_news=("ai_flag","mean"),
        pct_chip_news=("chip_flag","mean"),
        pct_reg_news=("reg_flag","mean"),
        news_volatility=("score","std"),
    ).reset_index()

    out["news_volatility"] = out["news_volatility"].fillna(0)

    # z-score
    out["senti_count_z"] = 0
    for tic in out["ticker"].unique():
        m = out["ticker"] == tic
        v = out.loc[m,"senti_count"]
        if len(v)>1 and v.std()>0:
            out.loc[m,"senti_count_z"] = (v - v.mean()) / v.std()

    # 1-day lag
    lag_cols = [
        "senti_pos","senti_neu","senti_neg","senti_score","senti_count",
        "pct_ai_news","pct_chip_news","pct_reg_news",
        "news_volatility","senti_count_z"
    ]
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker","date"])

    for tic in out["ticker"].unique():
        m = out["ticker"] == tic
        out.loc[m, lag_cols] = out.loc[m, lag_cols].shift(1)

    # neutral fill
    neutral = dict(
        senti_pos=1/3, senti_neu=1/3, senti_neg=1/3,
        senti_score=0, senti_count=0, pct_ai_news=0,
        pct_chip_news=0, pct_reg_news=0, news_volatility=0,
        senti_count_z=0
    )
    out.fillna(neutral, inplace=True)

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


# ----------------- MAIN -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_date", default="2023-01-01")
    ap.add_argument("--end_date",   default="2025-11-01")
    ap.add_argument("--output",     default="sentiment_ai_daily_multisource_streamed.csv")
    args = ap.parse_args()

    print("▶ Streaming HF dataset… (may take minutes)")
    recs = []

    for tic,date,head in stream_filtered_articles(args.start_date, args.end_date):
        lo = head.lower()
        recs.append(dict(
            ticker=tic,
            date=date,
            title=head,
            ai_flag   = any(w in lo for w in AI_WORDS),
            chip_flag = any(w in lo for w in CHIP_WORDS),
            reg_flag  = any(w in lo for w in REG_WORDS),
        ))

    print(f"✔ Matched article-ticker rows: {len(recs)}")

    if not recs:
        print("No rows found – check dates or dataset availability.")
        return

    print("▶ Running FinBERT…")
    df = build_daily_features(recs)

    print(f"✔ Daily rows: {len(df)}")
    df.to_csv(args.output, index=False)
    print(f"💾 Saved → {args.output}")


if __name__ == "__main__":
    main()

'''

python build_news_features_multisource_parquet.py ^
  --start_date 2023-01-01 ^
  --end_date   2025-11-01 ^
  --output sentiment_ai_daily_2023_2025_streamed.csv

'''