"""
Generate predictions on the holdout segment and run conservative backtests.
Saves per-ticker CSVs in reports/.
"""
import os
import pandas as pd
import torch
import joblib

from src.model import LSTMClassifier, make_sequences
from src.backtest import conservative_backtest

FEATURE_COLS = ["Return_1d","SMA_5","SMA_10","SMA_20","Vol_10","sentiment","sentiment_3d"]

def predict_for_ticker(df_t: pd.DataFrame, ticker: str, models_dir: str, lookback: int = 10):
    df_t = df_t.dropna(subset=FEATURE_COLS + ["target", "future_return_3d"]).copy()
    if len(df_t) < 200:
        raise RuntimeError(f"Not enough data for {ticker} after cleaning ({len(df_t)} rows).")
    n = len(df_t)
    split = int(n*0.8)
    test_df = df_t.iloc[split:].copy()
    scaler = joblib.load(os.path.join(models_dir, f"{ticker}_scaler.joblib"))
    model = LSTMClassifier(n_features=len(FEATURE_COLS))
    model.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker}_lstm.pt"), map_location="cpu"))
    model.eval()

    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
    X_test, y_test = make_sequences(test_df, FEATURE_COLS, "target", ticker=ticker, lookback=lookback)
    aligned = test_df.iloc[lookback:].copy()
    with torch.no_grad():
        pt = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
    aligned["prob"] = pt
    return aligned

def main(features_path: str, models_dir: str = "models", reports_dir: str = "reports", lookback: int = 10, threshold: float = 0.75, hold_days: int = 3):
    os.makedirs(reports_dir, exist_ok=True)
    df = pd.read_parquet(features_path)
    tickers = sorted(df["Ticker"].dropna().unique().tolist())
    for t in tickers:
        dft = df[df["Ticker"]==t].copy()
        try:
            pred_df = predict_for_ticker(dft, t, models_dir=models_dir, lookback=lookback)
            pred_df.to_csv(os.path.join(reports_dir, f"{t}_predictions.csv"), index=False)
            bt = conservative_backtest(pred_df, prob_col="prob", threshold=threshold, hold_days=hold_days, fee_bps=5)
            bt[["Date","Ticker","equity"]].to_csv(os.path.join(reports_dir, f"{t}_equity.csv"), index=False)
            print(f"[{t}] predictions & equity saved.")
        except Exception as e:
            print(f"[{t}] ERROR: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--hold_days", type=int, default=3)
    args = parser.parse_args()
    main(args.features, args.models_dir, args.reports_dir, lookback=args.lookback, threshold=args.threshold, hold_days=args.hold_days)
