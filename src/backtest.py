"""
Backtesting with conservative rules: position only if prob>threshold, max one open position, hold N days.
"""
import pandas as pd
import numpy as np

def conservative_backtest(pred_df: pd.DataFrame, prob_col: str = "prob", threshold: float = 0.75, hold_days: int = 3, fee_bps: float = 5):
    """
    pred_df requires columns: Date, Ticker, Close, prob, future_return_{hold_days}d
    fee_bps: round-trip fee in basis points (e.g., 5 = 0.05% per trade)
    """
    pred_df = pred_df.sort_values(["Date","Ticker"]).copy()
    pred_df["signal"] = (pred_df[prob_col] >= threshold).astype(int)
    equity = 1.0
    equity_curve = []
    for _, row in pred_df.iterrows():
        if row["signal"] == 1:
            gross = 1.0 + row[f"future_return_{hold_days}d"]
            net = gross * (1 - fee_bps/10000)
            equity *= net
        equity_curve.append(equity)
    pred_df["equity"] = equity_curve
    return pred_df
