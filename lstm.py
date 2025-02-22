import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# Configuration
INPUT_DIR = "data/walk_forward_splits_gap"
OUTPUT_FILE = "data/lstm_results.csv"
BATCH_SIZE = 256
EPOCHS = 15
DEVICE = torch.device("cpu")
SEQ_LENGTH = 20

# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_length=SEQ_LENGTH):
        self.seq_length = seq_length
        self.features = df.drop(columns=["timestamp", "label"]).values
        self.labels = df["label"].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels) - self.seq_length

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_length
        X = self.features[start_idx:end_idx]
        y = self.labels[end_idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)  # [batch_size, seq_length, hidden_size]
        out = self.dropout(out[:, -1, :])  # Last timestep
        out = self.fc(out)  # [batch_size, output_size]
        return out

# Training function with early stopping
def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 0.5]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "lstm_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_val_loss

# Main execution
def main():
    print("=== LSTM Training for HFT (0.5% TP, 0.2% SL, 1-Hour Horizon) ===")
    results = []

    # Load Stage 5 data for Window 0
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train_0.csv"), parse_dates=["timestamp"])
    val_df = pd.read_csv(os.path.join(INPUT_DIR, "val_0.csv"), parse_dates=["timestamp"])
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test_0.csv"), parse_dates=["timestamp"])

    # Prepare datasets
    train_dataset = TimeSeriesDataset(train_df)
    val_dataset = TimeSeriesDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    input_size = train_df.drop(columns=["timestamp", "label"]).shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=1)

    # Train
    best_val_loss = train_model(model, train_loader, val_loader, EPOCHS, DEVICE)
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Load best model and evaluate (simplified for Window 0)
    model.load_state_dict(torch.load("lstm_best.pth"))
    model.eval()
    test_dataset = TimeSeriesDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    # PnL simulation (adapted from TFT)
    closes = test_df["close"].values[-len(all_preds):]
    gains = []
    num_trades = 0
    for i, pred in enumerate(all_preds):
        if i >= len(closes) - 1 or pred == 2:
            gains.append(0.0)
            continue
        entry_price = closes[i]
        if pred == 1:  # Long
            tp_price = entry_price * 1.005
            sl_price = entry_price * 0.998
            future_prices = closes[i+1 : min(i+21, len(closes))]
            hit_tp = any(p >= tp_price for p in future_prices)
            hit_sl = any(p <= sl_price for p in future_prices)
            if hit_tp:
                gain = 0.005 - 0.002
                gains.append(gain)
            elif hit_sl:
                gain = -0.002 - 0.002
                gains.append(gain)
            else:
                gains.append(-0.002)
            num_trades += 1
        elif pred == 0:  # Short
            tp_price = entry_price * 0.995
            sl_price = entry_price * 1.002
            future_prices = closes[i+1 : min(i+21, len(closes))]
            hit_tp = any(p <= tp_price for p in future_prices)
            hit_sl = any(p >= sl_price for p in future_prices)
            if hit_tp:
                gain = 0.005 - 0.002
                gains.append(gain)
            elif hit_sl:
                gain = -0.002 - 0.002
                gains.append(gain)
            else:
                gains.append(-0.002)
            num_trades += 1
    
    total_pnl = sum(gains)
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0

    row = {
        "window_index": 0,
        "train_start": train_df["timestamp"].min(),
        "train_end": train_df["timestamp"].max(),
        "val_start": val_df["timestamp"].min(),
        "val_end": val_df["timestamp"].max(),
        "test_start": test_df["timestamp"].min(),
        "test_end": test_df["timestamp"].max(),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "val_loss": best_val_loss,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "avg_pnl_per_trade": avg_pnl,
    }
    results.append(row)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    print(df_results)

if __name__ == "__main__":
    main()