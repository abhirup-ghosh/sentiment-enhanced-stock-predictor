"""
Sentiment ingestion and scoring with NewsAPI + VADER.
"""
import os
import datetime as dt
import pandas as pd
import requests
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import List

# load .env only once
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

_COMPANY_BY_TICKER = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
}

def fetch_newsapi_headlines(ticker: str, start: str, end: str) -> List[str]:
    """Fetch recent headlines using NewsAPI free tier (last ~30 days)."""
    if NEWSAPI_KEY is None:
        return []

    company = 'Trump'#_COMPANY_BY_TICKER.get(ticker, ticker)
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "q": company,
        "language": "en",
        "pageSize": 100,
        "page": 1,
        "apiKey": NEWSAPI_KEY,
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"[WARN] NewsAPI request failed: {r.text}")
        return []
    data = r.json()
    return [a["title"] for a in data.get("articles", [])]

def fetch_yahoo_headlines(ticker: str) -> List[str]:
    """Fetch headlines from Yahoo Finance (works historically)."""
    import yfinance as yf
    try:
        news = yf.Ticker(ticker).news
        return [item.get("title", "") for item in news if "title" in item]
    except Exception as e:
        print(f"[WARN] Yahoo Finance news fetch failed for {ticker}: {e}")
        return []

def get_daily_sentiment(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Get daily average sentiment for a ticker."""
    headlines = fetch_newsapi_headlines(ticker, start, end)
    if not headlines:  # fallback if NewsAPI fails
        headlines = fetch_yahoo_headlines(ticker)

    scored = score_sentiment_vader(headlines)
    if not scored:
        return pd.DataFrame(columns=["Date", "sentiment", "Ticker"])

    df = pd.DataFrame(scored, columns=["Date", "sentiment"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.groupby("Date")["sentiment"].mean().reset_index()
    df["Ticker"] = ticker
    return df

def get_daily_sentiment(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Get daily average sentiment for a ticker."""
    headlines = fetch_newsapi_headlines(ticker, start, end)
    if not headlines:  # fallback if NewsAPI fails
        headlines = fetch_yahoo_headlines(ticker)

    scored = score_sentiment_vader(headlines)
    if not scored:
        return pd.DataFrame(columns=["Date", "sentiment", "Ticker"])

    df = pd.DataFrame(scored, columns=["Date", "sentiment"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.groupby("Date")["sentiment"].mean().reset_index()
    df["Ticker"] = ticker
    return df

def score_daily_sentiment_vader(headlines_df: pd.DataFrame) -> pd.DataFrame:
    if headlines_df is None or headlines_df.empty:
        return pd.DataFrame(columns=["date","sentiment"])
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    df = headlines_df.copy()
    df["compound"] = df["title"].fillna("").astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
    out = df.groupby("date", as_index=False)["compound"].mean().rename(columns={"compound": "sentiment"})
    return out

def get_daily_sentiment(ticker: str, start: str, end: str) -> pd.DataFrame:
    headlines = fetch_newsapi_headlines(ticker, start, end)
    headlines_df = pd.DataFrame({"title": headlines, "date": [dt.date.today()] * len(headlines)})
    daily = score_daily_sentiment_vader(headlines_df)
    daily["ticker"] = ticker
    return daily
