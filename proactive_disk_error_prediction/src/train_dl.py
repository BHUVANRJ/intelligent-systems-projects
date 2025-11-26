# src/train_dl.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import preprocess
from feature_engineering import add_features
from config import DATA_PATH, LABEL_COLUMN, SMART_FEATURES

# DEVICE CONFIGURATION

DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", DEVICE)


# MODEL FOLDER

MODEL_DIR = "../models/dl_models"
os.makedirs(MODEL_DIR, exist_ok=True)



# DATA PREPARATION

def build_sequences(df, sequence_length=10):
    """
    Build sequences for DL. Automatically detects a time column if present.
    Falls back to index ordering if no time column exists.
    """

    # Try to detect a time column
    possible_time_cols = ["date", "timestamp", "time", "datetime", "collect_time"]

    time_col = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    # Sort if time column exists
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col)
    else:
        print(" No time column found — using row index order.")
        df = df.reset_index(drop=True)

    # Sequence creation
    X, y = [], []
    features = SMART_FEATURES

    for i in range(len(df) - sequence_length):
        seq = df[features].iloc[i:i + sequence_length].values
        label = df[LABEL_COLUMN].iloc[i + sequence_length]

        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)



def get_dataloaders(batch_size=32, sequence_length=10):
    df = preprocess(DATA_PATH)
    df = add_features(df)

    X, y = build_sequences(df, sequence_length)

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, test_loader, X_train.shape[-1]



# MODELS


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))


class CNN1DModel(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear((sequence_length // 2) * 32, 1)

    def forward(self, x):
        # reshape for conv: (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.flatten(1)
        return torch.sigmoid(self.fc(x))



# TRAINING FUNCTION


def train(model, train_loader, test_loader, epochs=15, lr=0.001):
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.4f}")

    # ------ Evaluation ------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = (model(X_batch).squeeze() > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    accuracy = correct / total
    print(f"\n Final Test Accuracy: {accuracy:.4f}")

    return model



# RUN ALL MODELS


def train_all_dl_models():
    print("\nLoading data...")
    train_loader, test_loader, input_dim = get_dataloaders()
    seq_len = next(iter(train_loader))[0].shape[1]

    print("Training LSTM...")
    lstm = train(LSTMModel(input_dim), train_loader, test_loader)
    torch.save(lstm.state_dict(), f"{MODEL_DIR}/lstm_model.pt")

    print("Training GRU...")
    gru = train(GRUModel(input_dim), train_loader, test_loader)
    torch.save(gru.state_dict(), f"{MODEL_DIR}/gru_model.pt")

    print("Training 1D CNN...")
    cnn = train(CNN1DModel(input_dim, seq_len), train_loader, test_loader)
    torch.save(cnn.state_dict(), f"{MODEL_DIR}/cnn1d_model.pt")

    print("\n  All DL models trained and saved!")


if __name__ == "__main__":
    train_all_dl_models()
