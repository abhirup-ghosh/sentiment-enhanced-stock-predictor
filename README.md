# Sentiment-Enhanced Stock Predictor (MVP)

An end-to-end MVP that predicts short-term (3-day) stock direction using price + news sentiment with an LSTM model.
Designed for conservative trading: only trade when model confidence is high.

## Repo structure
```
sentiment-stock-predictor/
├── data/               # raw & processed data (gitignored)
├── notebooks/          # starter notebooks for EDA & experiments
├── src/                # source code: data, features, sentiment, train, predict
├── app/                # Streamlit dashboard
├── models/             # saved models and scalers
├── reports/            # predictions, metrics, equity curves
├── requirements.txt
├── README.md
└── LICENSE
```

## One-time setup
1. Create and activate a virtualenv (Python 3.10+ recommended).
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Add your NewsAPI key in `.env` in the project root:
```
NEWSAPI_KEY=your_key_here
```

## End-to-end usage
A) Build the dataset (prices + sentiment + features + targets)
```bash
python -m src.build_dataset --tickers AAPL,MSFT,NVDA --start 2023-01-01 --end 2025-09-01 --out data/processed/features.parquet
```

B) Train LSTM models (per ticker)
```bash
python -m src.train --features data/processed/features.parquet --lookback 10 --epochs 10
```

C) Generate predictions & conservative backtests
```bash
python -m src.predict_and_backtest --features data/processed/features.parquet --lookback 10 --threshold 0.75 --hold_days 3
```

D) Launch the dashboard
```bash
streamlit run app/dashboard.py
```

## Notes
- The sentiment ingestor uses NewsAPI + NLTK VADER (free tier).
- Conservative rules: only trade when `prob >= threshold` (default 0.75), limited holding period, and fees included.
- Check `reports/` for metrics and outputs.
