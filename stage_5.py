import pandas as pd
import numpy as np
import os

LABELED_DATA_PATH = "data/labeled_dynamic.csv"
OUTPUT_DIR = "data/walk_forward_splits_gap"

# Walk-forward configuration
TRAIN_MONTHS = 8
VAL_MONTHS   = 1
GAP_MONTHS   = 1  # additional gap to avoid overlap with test
TEST_MONTHS  = 1
SLIDE_MONTHS = 1  # how many months to shift for the next window

def main():
    print("=== Stage 4: Walk-Forward Splitting (8:1:1 ratio + 1-month gap) ===")
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1) Detect dataset start/end
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    print(f"Dataset date range: {min_date} to {max_date}, total rows = {len(df)}")

    # 2) Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Helper to add months via pandas
    def add_months(dt, n_months):
        return dt + pd.DateOffset(months=n_months)

    window_index = 0
    start_time = min_date

    while True:
        # 3) Define each segment
        # Train: 8 months
        train_start = start_time
        train_end   = add_months(train_start, TRAIN_MONTHS)

        # Validation: next 1 month
        val_start = train_end
        val_end   = add_months(val_start, VAL_MONTHS)

        # Gap: skip 1 month
        gap_start = val_end
        gap_end   = add_months(gap_start, GAP_MONTHS)

        # Test: next 1 month
        test_start = gap_end
        test_end   = add_months(test_start, TEST_MONTHS)

        # If test_start is beyond the dataset max, we stop
        if test_start > max_date:
            print("Reached beyond dataset's end date. Stopping.")
            break

        # 4) Filter data for each segment
        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        df_val   = df[(df["timestamp"] >= val_start)   & (df["timestamp"] < val_end)]
        # We do NOT produce a dataset for the gap, it's intentionally skipped
        df_test  = df[(df["timestamp"] >= test_start)  & (df["timestamp"] < test_end)]

        # Stop if the train set is empty (no point continuing)
        if len(df_train) == 0:
            print(f"No training data in window {window_index}. Ending splits.")
            break

        # 5) Validate that there's no overlap in the date ranges
        #    We'll print them out and do basic checks:
        def check_overlap(df_a, df_b, name_a, name_b):
            if len(df_a) == 0 or len(df_b) == 0:
                return False
            max_a = df_a["timestamp"].max()
            min_b = df_b["timestamp"].min()
            return max_a >= min_b  # indicates overlap
        overlap_train_val  = check_overlap(df_train, df_val, "train", "val")
        overlap_val_test   = check_overlap(df_val, df_test, "val", "test")
        overlap_train_test = check_overlap(df_train, df_test, "train", "test")

        if overlap_train_val or overlap_val_test or overlap_train_test:
            print(f"WARNING: Overlap detected in window {window_index}. "
                  "Check your date boundaries or reduce GAP_MONTHS.")
        else:
            print(f"No overlap detected for window {window_index}.")

        # 6) Log row counts and date ranges
        print(f"--- Window {window_index} ---")
        print(f"Train: {train_start.date()} to {train_end.date()} => {len(df_train)} rows")
        print(f"Valid: {val_start.date()}  to {val_end.date()}   => {len(df_val)} rows")
        print(f" Gap:  {gap_start.date()}  to {gap_end.date()}   => (skipped)")
        print(f" Test: {test_start.date()} to {test_end.date()}  => {len(df_test)} rows")

        # 7) Save CSVs
        train_file = os.path.join(OUTPUT_DIR, f"train_{window_index}.csv")
        val_file   = os.path.join(OUTPUT_DIR, f"val_{window_index}.csv")
        test_file  = os.path.join(OUTPUT_DIR, f"test_{window_index}.csv")
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)

        # 8) (Optional) Train a model & compute metrics here
        #    For demonstration, we'll just show placeholders.
        #    In a real scenario, you might call a function:
        #    metrics = train_and_evaluate_model(df_train, df_val, df_test)
        #    print(metrics)

        # 9) Slide forward
        start_time = add_months(start_time, SLIDE_MONTHS)
        window_index += 1

        if start_time > max_date:
            print("No further windows possible. Done.")
            break

if __name__ == "__main__":
    main()
