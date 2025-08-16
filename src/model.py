"""
LSTM model definition & simple training loop (PyTorch).
"""
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

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

def make_sequences(df: pd.DataFrame, feature_cols, target_col, ticker: str, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    d = df[df["Ticker"]==ticker].dropna(subset=feature_cols+[target_col]).copy()
    X, y = [], []
    vals = d[feature_cols].values.astype(np.float32)
    target = d[target_col].values.astype(np.float32)
    for i in range(len(d)-lookback):
        X.append(vals[i:i+lookback])
        y.append(target[i+lookback])
    return np.array(X), np.array(y)

def train_simple_lstm(X: np.ndarray, y: np.ndarray, epochs: int = 10, lr: float = 1e-3) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(n_features=X.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss = bce(pred, y_t)
        loss.backward()
        opt.step()
    return model
