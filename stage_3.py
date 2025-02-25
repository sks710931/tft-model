import pandas as pd
import numpy as np
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
import os

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, "stage_3.log"),
    maxBytes=1_000_000,  # 1MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

ENGINEERED_DATA_PATH = "data/engineered.csv"
LABELED_DATA_PATH = "data/labeled_dynamic.csv"

FUTURE_HORIZON = 15   # 45 minutes
ENTRY_FEE = 0.001     # 0.1%
EXIT_FEE = 0.001      # 0.1%
TOTAL_FEE = ENTRY_FEE + EXIT_FEE  # 0.2%
MIN_PROFIT = 0.001    # 0.1% profit to boost trades
FIXED_THRESHOLD = MIN_PROFIT + TOTAL_FEE  # 0.3%
TARGET_TRADE_PCT = 0.50  # 50% trades

def label_with_fixed_threshold(closes):
    """Label using a fixed profit threshold."""
    labels = np.full(len(closes), fill_value=2, dtype=int)
    
    for i in range(len(closes) - FUTURE_HORIZON):
        entry_price = closes[i]
        future_window = closes[i+1:i+1+FUTURE_HORIZON]
        if len(future_window) == 0:
            continue
        
        max_up = max((p / entry_price - 1) - TOTAL_FEE for p in future_window)
        max_down = max((1 - p / entry_price) - TOTAL_FEE for p in future_window)
        
        if max_up >= FIXED_THRESHOLD:
            labels[i] = 1  # Long
        elif max_down >= FIXED_THRESHOLD:
            labels[i] = 0  # Short
    
    return labels

def balance_labels(df, labels, closes, target_trade_pct=TARGET_TRADE_PCT):
    """Adjust labels to hit 50% trade percentage."""
    total_rows = len(df) - FUTURE_HORIZON
    target_trades = int(total_rows * target_trade_pct)
    current_trades = sum(1 for label in labels if label != 2)
    
    if current_trades < target_trades:
        shortfall = target_trades - current_trades
        no_trade_idx = [i for i in range(len(labels) - FUTURE_HORIZON) if labels[i] == 2]
        if no_trade_idx:
            candidates = sorted(
                [(i, max(max((closes[j] / closes[i] - 1) - TOTAL_FEE, 0) for j in range(i+1, i+1+FUTURE_HORIZON)) +
                      max(max((1 - closes[j] / closes[i]) - TOTAL_FEE, 0) for j in range(i+1, i+1+FUTURE_HORIZON)))
                 for i in no_trade_idx],
                key=lambda x: x[1],
                reverse=True
            )
            for i, _ in candidates[:shortfall]:
                max_up = max((closes[j] / closes[i] - 1) - TOTAL_FEE for j in range(i+1, i+1+FUTURE_HORIZON))
                max_down = max((1 - closes[j] / closes[i]) - TOTAL_FEE for j in range(i+1, i+1+FUTURE_HORIZON))
                if max_up >= MIN_PROFIT:
                    labels[i] = 1
                elif max_down >= MIN_PROFIT:
                    labels[i] = 0
                else:
                    labels[i] = 1 if max_up > max_down else 0  # Fallback to hit 50%
    
    return labels

def main():
    logger.info("=== Stage 3: Labeling for Near-HFT (Fixed Threshold 0.3%, 50% Trades, 45-Min Horizon, 0.2% Fees, 0.1% Min Profit) ===")
    
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=["timestamp"])
    initial_rows = len(df)
    logger.info(f"Initial dataset size: {initial_rows} rows")
    
    closes = df["close"].values
    dates = df["timestamp"].dt.date.values
    
    # Initial labeling
    labels = label_with_fixed_threshold(closes)
    
    # Balance to 50% trades
    labels = balance_labels(df, labels, closes)
    
    df["label"] = labels
    df = df.iloc[:-FUTURE_HORIZON].copy()
    
    # Log daily trade stats with explicit float conversion
    df["date"] = df["timestamp"].dt.date
    daily_trades = df[df["label"] != 2].groupby("date").size()
    logger.info(f"Daily trades - Mean: {float(daily_trades.mean()):.1f}, Median: {float(daily_trades.median()):.1f}, "
                f"Min: {float(daily_trades.min()):.1f}, Max: {float(daily_trades.max()):.1f}")
    
    label_counts = Counter(df["label"])
    total_labels = sum(label_counts.values())
    logger.info(f"Label distribution: {label_counts}")
    logger.info(f"Short trades (0): {label_counts[0]} ({label_counts[0]/total_labels:.2%})")
    logger.info(f"Long trades (1): {label_counts[1]} ({label_counts[1]/total_labels:.2%})")
    logger.info(f"No trade (2): {label_counts[2]} ({label_counts[2]/total_labels:.2%})")
    
    final_rows = len(df)
    logger.info(f"Final dataset rows: {final_rows}")
    logger.info(f"% Data lost: {100 * (initial_rows - final_rows) / initial_rows:.2f}%")
    
    df.drop(columns=["date"], inplace=True)
    df.to_csv(LABELED_DATA_PATH, index=False)
    logger.info(f"Labeled data saved to {LABELED_DATA_PATH}")

if __name__ == "__main__":
    main()