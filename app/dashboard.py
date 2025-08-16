import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sentiment-Enhanced Stock Predictor", layout="wide")

st.title("📈 Sentiment-Enhanced Stock Predictor (MVP)")
st.markdown("Conservative signals using price + sentiment.")

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", ["AAPL","MSFT","NVDA","TSLA","AMZN"])
    threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.01)
    hold_days = st.selectbox("Hold Days", [3,5])
    st.caption("This MVP shows placeholder outputs until model training is added.")

# Placeholder demo dataframe
dates = pd.date_range("2024-01-01", periods=120, freq="B")
close = np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates))) * 100
prob = np.clip(np.random.normal(0.6, 0.15, len(dates)), 0, 1)
df = pd.DataFrame({"Date": dates, "Close": close, "prob": prob})
df["signal"] = (df["prob"] >= threshold).astype(int)
df["future_return_3d"] = df["Close"].pct_change(3).shift(-3).fillna(0)
equity = 1.0
equity_curve = []
fee_bps = 5
for _, r in df.iterrows():
    if r["signal"] == 1:
        gross = 1.0 + r["future_return_3d"]
        net = gross * (1 - fee_bps/10000)
        equity *= net
    equity_curve.append(equity)
df["equity"] = equity_curve

tab1, tab2 = st.tabs(["Chart","Backtest"])

with tab1:
    st.line_chart(df.set_index("Date")[["Close"]])
    st.bar_chart(df.set_index("Date")[["prob"]])

with tab2:
    st.subheader("Equity Curve (placeholder)")
    st.line_chart(df.set_index("Date")[["equity"]])
    st.dataframe(df.tail(20))
