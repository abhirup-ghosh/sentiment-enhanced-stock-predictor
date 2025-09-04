"""
Feature engineering: technical indicators and rolling sentiment features.
"""
import pandas as pd
import numpy as np

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker","Date"])
    df["Return_1d"] = df.groupby("Ticker")["Close"].pct_change()
    for w in [5,10,20]:
        df[f"SMA_{w}"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(w).mean())
    df["Vol_10"] = df.groupby("Ticker")["Return_1d"].transform(lambda s: s.rolling(10).std())
    return df

def merge_sentiment(price_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    p = price_df.copy()
    s = sent_df.copy()
    s["date"] = pd.to_datetime(s["date"])
    p["date"] = pd.to_datetime(p["Date"])
    feat = p.merge(s, how="left", left_on=["date","Ticker"], right_on=["date","ticker"])
    feat["sentiment"] = feat["sentiment"].fillna(0.0)
    feat["sentiment_3d"] = feat.groupby("Ticker")["sentiment"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    return feat
