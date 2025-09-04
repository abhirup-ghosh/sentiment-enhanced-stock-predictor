"""
LSTM model definition & simple training utilities (PyTorch).
"""
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out,_ = self.lstm(x)
        last = out[:,-1,:]
        return self.head(last).squeeze(-1)

def make_sequences(df: pd.DataFrame, feature_cols, target_col, ticker: str, lookback: int = 10):
    d = df[df["Ticker"]==ticker].dropna(subset=feature_cols+[target_col]).copy()
    X, y = [], []
    vals = d[feature_cols].values.astype(float)
    target = d[target_col].values.astype(float)
    for i in range(len(d)-lookback):
        X.append(vals[i:i+lookback])
        y.append(target[i+lookback])
    return np.array(X), np.array(y)
