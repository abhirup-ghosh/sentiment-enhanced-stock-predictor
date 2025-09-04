"""
Backtesting with conservative rules: position only if prob>threshold, simple equity curve.
"""
import pandas as pd

def conservative_backtest(pred_df: pd.DataFrame, prob_col: str = "prob", threshold: float = 0.75, hold_days: int = 3, fee_bps: float = 5):
    pred_df = pred_df.sort_values(["Date","Ticker"]).copy()
    pred_df["signal"] = (pred_df[prob_col] >= threshold).astype(int)
    equity = 1.0
    equity_curve = []
    for _, row in pred_df.iterrows():
        if row["signal"] == 1:
            gross = 1.0 + row.get(f"future_return_{hold_days}d", 0.0)
            net = gross * (1 - fee_bps/10000)
            equity *= net
        equity_curve.append(equity)
    pred_df["equity"] = equity_curve
    return pred_df
