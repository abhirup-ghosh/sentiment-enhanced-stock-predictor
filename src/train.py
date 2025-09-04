"""
Train an LSTM classifier per ticker on the processed dataset.
Saves models and reports.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib

from src.model import LSTMClassifier, make_sequences

FEATURE_COLS = ["Return_1d","SMA_5","SMA_10","SMA_20","Vol_10","sentiment","sentiment_3d"]

def time_split(df: pd.DataFrame, split_ratio: float = 0.8):
    n = len(df)
    split = int(n * split_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def train_for_ticker(df_t: pd.DataFrame, ticker: str, lookback: int = 10, epochs: int = 10, lr: float = 1e-3):
    df_t = df_t.dropna(subset=FEATURE_COLS + ["target"]).copy()
    if len(df_t) < 200:
        raise RuntimeError(f"Not enough data for {ticker} after cleaning ({len(df_t)} rows).")
    train_df, val_df = time_split(df_t, split_ratio=0.8)
    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])

    X_train, y_train = make_sequences(train_df, FEATURE_COLS, "target", ticker=ticker, lookback=lookback)
    X_val, y_val = make_sequences(val_df, FEATURE_COLS, "target", ticker=ticker, lookback=lookback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(n_features=X_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss = bce(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy()
    yv_np = y_val
    metrics = {
        "accuracy": float(accuracy_score(yv_np, (pv>=0.5).astype(int))),
        "precision": float(precision_score(yv_np, (pv>=0.5).astype(int), zero_division=0)),
        "recall": float(recall_score(yv_np, (pv>=0.5).astype(int), zero_division=0)),
        "auc": float(roc_auc_score(yv_np, pv)) if len(np.unique(yv_np))>1 else None,
        "n_val": int(len(yv_np)),
    }
    return model, scaler, metrics

def main(features_path: str, out_models: str = "models", reports_dir: str = "reports", lookback: int = 10, epochs: int = 10):
    os.makedirs(out_models, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    df = pd.read_parquet(features_path)
    tickers = sorted(df["Ticker"].dropna().unique().tolist())
    results = {}
    for t in tickers:
        dft = df[df["Ticker"]==t].copy()
        try:
            model, scaler, metrics = train_for_ticker(dft, t, lookback=lookback, epochs=epochs)
            torch.save(model.state_dict(), os.path.join(out_models, f"{t}_lstm.pt"))
            joblib.dump(scaler, os.path.join(out_models, f"{t}_scaler.joblib"))
            with open(os.path.join(reports_dir, f"{t}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            results[t] = metrics
            print(f"[{t}] {metrics}")
        except Exception as e:
            print(f"[{t}] ERROR: {e}")
    with open(os.path.join(reports_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.features, args.models_dir, args.reports_dir, lookback=args.lookback, epochs=args.epochs)
