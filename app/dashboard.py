import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Sentiment-Enhanced Stock Predictor", layout="wide")
st.title("📈 Sentiment-Enhanced Stock Predictor (MVP)")
st.markdown("Conservative signals using price + sentiment.")

AVAILABLE_TICKERS = ["AAPL","MSFT","NVDA","TSLA","AMZN"]

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", AVAILABLE_TICKERS)
    threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.01)
    hold_days = st.selectbox("Hold Days", [3,5])
    st.caption("If predictions aren't found yet, the app will show placeholders.")

pred_path = f"reports/{ticker}_predictions.csv"
equity_path = f"reports/{ticker}_equity.csv"

if os.path.exists(pred_path) and os.path.exists(equity_path):
    st.success("Loaded real predictions & backtest results.")
    pred_df = pd.read_csv(pred_path, parse_dates=["Date"])
    eq_df = pd.read_csv(equity_path, parse_dates=["Date"])
    pred_df["signal"] = (pred_df["prob"] >= threshold).astype(int)

    tab1, tab2, tab3 = st.tabs(["Chart","Probabilities","Backtest"])
    with tab1:
        st.subheader(f"Price: {ticker}")
        st.line_chart(pred_df.set_index("Date")[["Close"]])
    with tab2:
        st.subheader("Predicted Probability (Up in next 3 days)")
        st.bar_chart(pred_df.set_index("Date")[["prob"]])
    with tab3:
        st.subheader("Equity Curve")
        st.line_chart(eq_df.set_index("Date")[["equity"]])
        st.dataframe(pred_df[["Date","Ticker","Close","prob","signal","future_return_3d"]].tail(30))
else:
    st.warning("No real predictions found in reports/. Showing placeholders.")
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

    tab1, tab2, tab3 = st.tabs(["Chart","Probabilities","Backtest"])
    with tab1:
        st.line_chart(df.set_index("Date")[["Close"]])
    with tab2:
        st.bar_chart(df.set_index("Date")[["prob"]])
    with tab3:
        st.subheader("Equity Curve (placeholder)")
        st.line_chart(df.set_index("Date")[["equity"]])
        st.dataframe(df.tail(20))
