import pandas as pd
import numpy as np
import os

LABELED_DATA_PATH = "data/labeled_dynamic.csv"
OUTPUT_DIR = "data/walk_forward_splits_gap"

# Walk-forward configuration
TRAIN_MONTHS = 8      # 8 months (~243 days)
VAL_DAYS = 14         # 2 weeks = 14 days
GAP_DAYS = 14         # 2 weeks = 14 days
TEST_DAYS = 14        # 2 weeks = 14 days
SLIDE_DAYS = 14       # Slide by 2 weeks = 14 days

def main():
    print("=== Stage 5: Walk-Forward Splitting (8 Months Train, 2-Week Val/Test) ===")
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Detect dataset start/end
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    print(f"Dataset range: {min_date} to {max_date}, total rows = {len(df)}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Helper functions for date offsets
    def add_months(dt, n_months):
        return dt + pd.DateOffset(months=n_months)
    
    def add_days(dt, n_days):
        return dt + pd.offsets.Day(n_days)

    window_index = 0
    start_time = min_date

    while True:
        # Define each segment
        train_start = start_time
        train_end = add_months(train_start, TRAIN_MONTHS)
        val_start = train_end
        val_end = add_days(val_start, VAL_DAYS)
        gap_start = val_end
        gap_end = add_days(gap_start, GAP_DAYS)
        test_start = gap_end
        test_end = add_days(test_start, TEST_DAYS)

        # Stop if test_start exceeds dataset max
        if test_start > max_date:
            print("Reached beyond dataset's end date. Stopping.")
            break

        # Filter data for each segment
        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        df_val = df[(df["timestamp"] >= val_start) & (df["timestamp"] < val_end)]
        df_test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)]

        # Stop if train set is empty
        if len(df_train) == 0:
            print(f"No training data in window {window_index}. Ending splits.")
            break

        # Corrected overlap check
        overlap_train_val = df_train["timestamp"].max() >= val_end if len(df_train) > 0 and len(df_val) > 0 else False
        overlap_val_test = df_val["timestamp"].max() >= test_end if len(df_val) > 0 and len(df_test) > 0 else False
        overlap_train_test = df_train["timestamp"].max() >= test_end if len(df_train) > 0 and len(df_test) > 0 else False

        if overlap_train_val or overlap_val_test or overlap_train_test:
            print(f"WARNING: True overlap detected in window {window_index}. Check boundaries.")
        else:
            print(f"No overlap detected for window {window_index}.")

        # Log row counts and date ranges
        print(f"--- Window {window_index} ---")
        print(f"Train: {train_start.date()} to {train_end.date()} => {len(df_train)} rows")
        print(f"Valid: {val_start.date()} to {val_end.date()} => {len(df_val)} rows")
        print(f"Gap: {gap_start.date()} to {gap_end.date()} => (skipped)")
        print(f"Test: {test_start.date()} to {test_end.date()} => {len(df_test)} rows")

        # Save CSVs
        train_file = os.path.join(OUTPUT_DIR, f"train_{window_index}.csv")
        val_file = os.path.join(OUTPUT_DIR, f"val_{window_index}.csv")
        test_file = os.path.join(OUTPUT_DIR, f"test_{window_index}.csv")
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)

        # Slide forward
        start_time = add_days(start_time, SLIDE_DAYS)
        window_index += 1

        if start_time > max_date:
            print("No further windows possible. Done.")
            break

if __name__ == "__main__":
    main()