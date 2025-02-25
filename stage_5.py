import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
from collections import Counter
import numpy as np

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, "stage_5.log"),
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

LABELED_DATA_PATH = "data/labeled_dynamic.csv"
OUTPUT_DIR = "data/walk_forward_splits_gap"

TRAIN_DAYS = 28    # 4 weeks
VAL_DAYS = 5       # 5 days
GAP_DAYS = 3       # 3 days
TEST_DAYS = 5      # 5 days
SLIDE_DAYS = 13    # Val + Gap + Test
TOTAL_FEE = 0.002  # 0.2%
MIN_PROFIT = 0.001 # 0.1% from Stage 3
TARGET_TRADE_PCT = 0.50  # 50% trades per split

def balance_split_labels(df_split, target_trade_pct=TARGET_TRADE_PCT):
    """Balance labels in a split to target trade percentage."""
    total_rows = len(df_split)
    target_trades = int(total_rows * target_trade_pct)
    current_trades = len(df_split[df_split["label"] != 2])
    
    df_split = df_split.copy()
    if current_trades > target_trades:
        excess = current_trades - target_trades
        trade_idx = df_split[df_split["label"] != 2].index
        drop_idx = np.random.choice(trade_idx, excess, replace=False)
        df_split.loc[drop_idx, "label"] = 2
    elif current_trades < target_trades:
        shortfall = target_trades - current_trades
        no_trade_idx = df_split[df_split["label"] == 2].index
        if len(no_trade_idx) > shortfall:
            candidates = df_split.loc[no_trade_idx].copy()
            candidates["max_up"] = candidates["future_close"] / candidates["close"] - 1 - TOTAL_FEE
            candidates["max_down"] = 1 - candidates["future_close"] / candidates["close"] - TOTAL_FEE
            candidates["score"] = np.maximum(candidates["max_up"], candidates["max_down"])
            top_idx = candidates.nlargest(shortfall, "score").index
            for i in top_idx:
                if candidates.loc[i, "max_up"] >= MIN_PROFIT:
                    df_split.loc[i, "label"] = 1
                elif candidates.loc[i, "max_down"] >= MIN_PROFIT:
                    df_split.loc[i, "label"] = 0
    
    return df_split

def log_split_stats(df_split, split_name):
    label_counts = Counter(df_split["label"])
    total = len(df_split)
    logger.info(f"{split_name} labels: {label_counts}")
    logger.info(f"{split_name} trade %: {((label_counts[0] + label_counts[1]) / total):.2%}")
    if "profit" in df_split.columns and split_name != "Gap":
        profits = df_split[df_split["label"] != 2]["profit"]
        logger.info(f"{split_name} mean profit: {float(profits.mean()):.4f}")
        logger.info(f"{split_name} % ≥ 0.2%: {float((profits >= 0.002).mean()):.2%}")

def main():
    logger.info("=== Stage 5: Walk-Forward Splitting (4 Weeks Train, 5-Day Val/Test, 3-Day Gap) ===")
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Verify full dataset labels
    label_counts = Counter(df["label"])
    total = len(df)
    logger.info(f"Full dataset labels: {label_counts}")
    logger.info(f"Full dataset trade %: {((label_counts[0] + label_counts[1]) / total):.2%}")

    # Precompute profits
    df["future_close"] = df["close"].shift(-15)
    df["profit"] = pd.NA
    df.loc[df["label"] == 1, "profit"] = (df["future_close"] / df["close"] - 1) - TOTAL_FEE
    df.loc[df["label"] == 0, "profit"] = (1 - df["future_close"] / df["close"]) - TOTAL_FEE

    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    logger.info(f"Dataset range: {min_date} to {max_date}, total rows = {len(df)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def add_days(dt, n_days):
        return dt + pd.offsets.Day(n_days)

    window_index = 0
    start_time = min_date

    while True:
        train_start = start_time
        train_end = add_days(train_start, TRAIN_DAYS)
        val_start = train_end
        val_end = add_days(val_start, VAL_DAYS)
        gap_start = val_end
        gap_end = add_days(gap_start, GAP_DAYS)
        test_start = gap_end
        test_end = add_days(test_start, TEST_DAYS)

        if test_end > max_date:
            logger.info("Test end exceeds dataset max date. Stopping.")
            break

        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        df_val = df[(df["timestamp"] >= val_start) & (df["timestamp"] < val_end)]
        df_gap = df[(df["timestamp"] >= gap_start) & (df["timestamp"] < gap_end)]
        df_test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)]

        if len(df_train) == 0:
            logger.info(f"No training data in window {window_index}. Ending splits.")
            break

        # Balance each split to 50%
        df_train = balance_split_labels(df_train)
        df_val = balance_split_labels(df_val)
        df_test = balance_split_labels(df_test)

        logger.info(f"--- Window {window_index} ---")
        logger.info(f"Train: {train_start.date()} to {train_end.date()} => {len(df_train)} rows")
        log_split_stats(df_train, "Train")
        logger.info(f"Valid: {val_start.date()} to {val_end.date()} => {len(df_val)} rows")
        log_split_stats(df_val, "Valid")
        logger.info(f"Gap: {gap_start.date()} to {gap_end.date()} => {len(df_gap)} rows")
        log_split_stats(df_gap, "Gap")
        logger.info(f"Test: {test_start.date()} to {test_end.date()} => {len(df_test)} rows")
        log_split_stats(df_test, "Test")

        train_file = os.path.join(OUTPUT_DIR, f"train_{window_index}.csv")
        val_file = os.path.join(OUTPUT_DIR, f"val_{window_index}.csv")
        test_file = os.path.join(OUTPUT_DIR, f"test_{window_index}.csv")
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)

        start_time = add_days(start_time, SLIDE_DAYS)
        window_index += 1

    logger.info(f"Completed {window_index} windows.")

if __name__ == "__main__":
    main()