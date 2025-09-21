"""
Feature engineering: technical indicators and rolling sentiment features.
"""

import pandas as pd
import numpy as np

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the price dataframe:
    - Daily returns
    - Simple moving averages (SMA) for 5, 10, 20 days
    - 10-day rolling volatility of returns
    """
    df = df.copy()
    # Ensure data is sorted for rolling calculations
    df = df.sort_values(["Ticker","Date"])
    # Calculate daily returns
    df["Return_1d"] = df.groupby("Ticker")["Close"].pct_change()
    # Calculate SMAs for different windows
    for w in [5,10,20]:
        df[f"SMA_{w}"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(w).mean())
    # Calculate 10-day rolling volatility of returns
    df["Vol_10"] = df.groupby("Ticker")["Return_1d"].transform(lambda s: s.rolling(10).std())
    return df

def merge_sentiment(price_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment scores into the price dataframe:
    - Aligns on date and ticker
    - Fills missing sentiment with 0.0
    - Adds a 3-day rolling mean of sentiment as a feature
    """
    p = price_df.copy()
    s = sent_df.copy()
    # Ensure date columns are datetime for merging
    # If sentiment date is a weekend, shift it to previous Friday
    s["date"] = pd.to_datetime(s["date"])
    s["weekday"] = s["date"].dt.weekday
    s.loc[s["weekday"] == 5, "date"] = s.loc[s["weekday"] == 5, "date"] - pd.Timedelta(days=1)  # Saturday -> Friday
    s.loc[s["weekday"] == 6, "date"] = s.loc[s["weekday"] == 6, "date"] - pd.Timedelta(days=2)  # Sunday -> Friday
    s = s.drop(columns="weekday")
    p["date"] = pd.to_datetime(p["Date"])
    # Merge sentiment into price data
    feat = p.merge(s, how="left", left_on=["date","Ticker"], right_on=["date","ticker"])
    # Fill missing sentiment values
    feat["sentiment"] = feat["sentiment"].fillna(0.0)
    # Add 3-day rolling average of sentiment
    feat["sentiment_3d"] = feat.groupby("Ticker")["sentiment"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    return feat
