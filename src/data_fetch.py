"""
Data fetching utilities: price data via yfinance, news headlines via NewsAPI.
"""
from typing import List
import pandas as pd
import yfinance as yf
import os
import datetime as dt

def fetch_prices(tickers: List[str], start: str = "2022-01-01", end: str | None = None) -> pd.DataFrame:
    end = end or dt.date.today().isoformat()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # If multiple tickers, yfinance returns multiindex columns; standardize to flat columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1).rename_axis(["Date","Ticker"]).reset_index()
    else:
        df["Ticker"] = tickers[0]
        df = df.reset_index()
    return df

def dummy_fetch_news_sentiment(dates: List[pd.Timestamp], ticker: str) -> pd.DataFrame:
    """
    Placeholder that returns neutral sentiment. Replace with NewsAPI or scraping.
    Returns columns: ['date','ticker','sentiment']
    """
    s = pd.DataFrame({
        "date": pd.to_datetime(dates).date,
        "ticker": ticker,
        "sentiment": 0.0
    })
    return s
