# Sentiment-Enhanced Stock Predictor

An MVP stock movement prediction system combining price action with news sentiment, powered by LSTMs.
Designed for conservative trading strategies with limited capital.

## Features
- Fetches OHLCV data for selected stocks
- Scrapes daily news headlines & computes sentiment
- Generates technical + sentiment features
- Trains LSTM to predict 3-day price movement
- Backtests strategy with position sizing & confidence threshold
- Streamlit dashboard for visualization

## Tech Stack
- Python 3.10+
- Pandas, NumPy
- yfinance
- NewsAPI / Yahoo Finance scraper
- NLTK (VADER) / FinBERT (later)
- PyTorch (LSTM)
- Streamlit

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-stock-predictor.git
cd sentiment-stock-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Run the Streamlit app:
```bash
streamlit run app/dashboard.py
```

## Roadmap
1. MVP with 3–5 stocks & daily data
2. Improve sentiment modeling with FinBERT
3. Add advanced risk management
4. Automate data updates & retraining
