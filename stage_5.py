import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler

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

TRAIN_DAYS = 14    # 2 weeks
VAL_DAYS = 5       # 5 days
GAP_DAYS = 3       # 3 days
TEST_DAYS = 5      # 5 days
SLIDE_DAYS = 13    # 5 val + 3 gap + 5 test
OVERLAP_DAYS = 3   # 3-day overlap with training

def main():
    logger.info("=== Stage 5: Walk-Forward Splitting (2 Weeks Train + 3-Day Overlap, 5-Day Val/Test) ===")
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

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
        val_start = add_days(train_end, -OVERLAP_DAYS)
        val_end = add_days(val_start, VAL_DAYS)
        gap_start = val_end
        gap_end = add_days(gap_start, GAP_DAYS)
        test_start = gap_end
        test_end = add_days(test_start, TEST_DAYS)

        if test_start > max_date:
            logger.info("Reached beyond dataset's end date. Stopping.")
            break

        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        df_val = df[(df["timestamp"] >= val_start) & (df["timestamp"] < val_end)]
        df_test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)]

        if len(df_train) == 0:
            logger.info(f"No training data in window {window_index}. Ending splits.")
            break

        logger.info(f"--- Window {window_index} ---")
        logger.info(f"Train: {train_start.date()} to {train_end.date()} => {len(df_train)} rows")
        logger.info(f"Valid: {val_start.date()} to {val_end.date()} => {len(df_val)} rows")
        logger.info(f"Gap: {gap_start.date()} to {gap_end.date()} => (skipped)")
        logger.info(f"Test: {test_start.date()} to {test_end.date()} => {len(df_test)} rows")

        train_file = os.path.join(OUTPUT_DIR, f"train_{window_index}.csv")
        val_file = os.path.join(OUTPUT_DIR, f"val_{window_index}.csv")
        test_file = os.path.join(OUTPUT_DIR, f"test_{window_index}.csv")
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)

        start_time = add_days(start_time, SLIDE_DAYS)
        window_index += 1

        if start_time > max_date:
            logger.info("No further windows possible. Done.")
            break

if __name__ == "__main__":
    main()