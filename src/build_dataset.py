"""
Builds the training dataset end-to-end:
- Fetch prices for tickers
- Fetch daily sentiment via NewsAPI + VADER
- Merge features
- Create targets (future_return_3d) and binary target
Saves processed dataset to data/processed/features.parquet
"""
import os
import argparse
import datetime as dt
import pandas as pd

from src.data_fetch import fetch_prices
from src.features import add_technical_features, merge_sentiment
from src.sentiment import get_daily_sentiment

def create_targets(df: pd.DataFrame, hold_days: int = 3) -> pd.DataFrame:
    """
    Adds future return and binary target columns to the dataframe.
    - future_return_{hold_days}d: percent change in Close price over hold_days, shifted to align with prediction date
    - target: binary indicator (1 if future return > 0, else 0)
    """
    df = df.sort_values(["Ticker","Date"]).copy()
    df[f"future_return_{hold_days}d"] = df.groupby("Ticker")["Close"].pct_change(hold_days).shift(-hold_days)
    df["target"] = (df[f"future_return_{hold_days}d"] > 0).astype(int)
    return df

def main(tickers, start, end, out_path):
    """
    Main workflow to build the dataset:
    - Fetch price data for all tickers
    - Fetch and combine daily sentiment for each ticker
    - Add technical features
    - Merge sentiment features
    - Add target columns
    - Save processed features to disk
    """
    prices = fetch_prices(tickers, start=start, end=end)
    all_sent = []
    for t in tickers:
        s = get_daily_sentiment(t, start, end)
        all_sent.append(s)
    sent = pd.concat(all_sent, ignore_index=True) if all_sent else pd.DataFrame(columns=["date","sentiment","ticker"])
    feat = add_technical_features(prices)
    feat = merge_sentiment(feat, sent)
    feat = create_targets(feat, hold_days=3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path} with shape {feat.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default=dt.date.today().isoformat())
    parser.add_argument("--out", type=str, default="data/processed/features.parquet")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    main(tickers, args.start, args.end, args.out)
