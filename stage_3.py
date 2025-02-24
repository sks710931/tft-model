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

FUTURE_HORIZON = 15   # 45 minutes (increased from 10)
ENTRY_FEE = 0.001     # 0.1%
EXIT_FEE = 0.001      # 0.1%
TOTAL_FEE = ENTRY_FEE + EXIT_FEE  # 0.2%
MIN_PROFIT = 0.002    # 0.2% minimum profit

def calibrate_threshold(df, closes, target_trade_pct=0.25):
    """Calibrate threshold to achieve ~50% trade labels, ensuring 0.2% profit."""
    movements = []
    for i in range(len(closes) - FUTURE_HORIZON):
        entry_price = closes[i]
        future_window = closes[i+1:i+1+FUTURE_HORIZON]
        if future_window.size > 0:
            up_move = max((p / entry_price - 1) for p in future_window)
            down_move = max((1 - p / entry_price) for p in future_window)
            movements.extend([up_move, down_move])
    
    movements = np.array(movements)
    valid_movements = movements[movements > TOTAL_FEE + MIN_PROFIT]  # Filter moves that meet profit requirement
    if len(valid_movements) == 0:
        logger.warning("No movements exceed minimum profit requirement. Using fallback threshold.")
        return TOTAL_FEE + MIN_PROFIT
    
    # Target 25th percentile to get ~50% trades after directionality
    threshold = np.percentile(valid_movements, target_trade_pct * 100)
    logger.info(f"Calibrated threshold: {threshold:.4f} to achieve ~{target_trade_pct*100}% potential trades")
    return threshold

def main():
    logger.info("=== Stage 3: Labeling for Near-HFT (Volatility-Adjusted, 45-Min Horizon, 0.2% Fees, 0.2% Min Profit) ===")
    
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=["timestamp"])
    initial_rows = len(df)
    logger.info(f"Initial dataset size: {initial_rows} rows")
    
    closes = df["close"].values
    atr_5 = df["atr_5"].values
    labels = np.full(len(df), fill_value=2, dtype=int)
    
    # Calibrate threshold
    dynamic_threshold = calibrate_threshold(df, closes, target_trade_pct=0.25)
    
    for i in range(len(df) - FUTURE_HORIZON):
        entry_price = closes[i]
        volatility_factor = atr_5[i] / atr_5.mean()
        movement_threshold = dynamic_threshold * max(0.5, min(2.0, volatility_factor))
        future_window = closes[i+1:i+1+FUTURE_HORIZON]
        
        if len(future_window) == 0:
            continue
        
        for price in future_window:
            up_movement = (price / entry_price - 1) - TOTAL_FEE
            down_movement = (1 - price / entry_price) - TOTAL_FEE
            if up_movement >= movement_threshold - TOTAL_FEE:
                labels[i] = 1
                break
            elif down_movement >= movement_threshold - TOTAL_FEE:
                labels[i] = 0
                break
    
    df["label"] = labels
    df = df.iloc[:-FUTURE_HORIZON].copy()
    
    label_counts = Counter(df["label"])
    total_labels = sum(label_counts.values())
    logger.info(f"Label distribution: {label_counts}")
    logger.info(f"Short trades (0): {label_counts[0]} ({label_counts[0]/total_labels:.2%})")
    logger.info(f"Long trades (1): {label_counts[1]} ({label_counts[1]/total_labels:.2%})")
    logger.info(f"No trade (2): {label_counts[2]} ({label_counts[2]/total_labels:.2%})")
    
    final_rows = len(df)
    logger.info(f"Final dataset rows: {final_rows}")
    logger.info(f"% Data lost: {100 * (initial_rows - final_rows) / initial_rows:.2f}%")
    
    df.to_csv(LABELED_DATA_PATH, index=False)
    logger.info(f"Labeled data saved to {LABELED_DATA_PATH}")

if __name__ == "__main__":
    main()