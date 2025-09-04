"""
Data fetching utilities: price data via yfinance.
"""
from typing import List
import pandas as pd
import yfinance as yf
import datetime as dt

def fetch_prices(tickers: List[str], start: str = "2022-01-01", end: str | None = None) -> pd.DataFrame:
    end = end or dt.date.today().isoformat()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # Normalize output to DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1).rename_axis(["Date","Ticker"]).reset_index()
        # columns e.g., ['Date','Ticker','Open','High',...]
    else:
        df = df.reset_index()
        # single ticker case: add Ticker column
        df["Ticker"] = tickers[0]
    # Ensure Date column is present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df
