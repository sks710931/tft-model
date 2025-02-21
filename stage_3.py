import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import resample

ENGINEERED_DATA_PATH = "data/engineered.csv"
LABELED_DATA_PATH = "data/labeled_dynamic.csv"

# Number of bars to look ahead
FUTURE_HORIZON = 10  # e.g., next 10 bars (~30 minutes if 3-min bars)

# Multiplier for ATR-based dynamic stop and target
SL_MULTIPLIER = 1.0   # e.g. 1 ATR for stop
TP_MULTIPLIER = 1.5   # e.g. 1.5 ATR for take-profit

# Slippage offset: dynamic approach
# We'll ensure at least a fraction of entry price, or some portion of ATR.
SLIPPAGE_OFFSET = 0.001  # 0.1% of entry_price
SLIPPAGE_ATR_FRAC = 0.1  # at least 10% of ATR

# Class imbalance threshold: if one label is more than THRESH times bigger than the other,
# we consider applying undersampling.
IMBALANCE_THRESHOLD = 2.0  # e.g. 2 => 2:1 ratio

def main():
    print("=== Stage 3: Dynamic Labeling with Logging & Class Imbalance Check ===")

    # 1) Load engineered data
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=["timestamp"])
    initial_rows = len(df)
    print(f"Initial dataset size: {initial_rows} rows")

    # Check we have an ATR column
    if "atr_14" not in df.columns:
        raise ValueError("ATR column (atr_14) not found. Please compute ATR in Stage 2.")

    closes = df["close"].values
    atrs = df["atr_14"].values

    # We'll store labels in an array, default to -1
    labels = np.full(len(df), fill_value=-1, dtype=int)

    # Counters for logging
    skip_atr_fallback_count = 0    # how many rows we can't label due to 0 ATR fallback
    constant_price_count = 0       # how many rows had constant future window
    labeled_count = 0              # how many rows actually labeled 0 or 1

    # 2) Loop to define dynamic TP & SL
    for i in range(len(df) - FUTURE_HORIZON):
        entry_price = closes[i]
        atr_now = atrs[i]

        # (A) Graceful handle ATR <= 0
        if atr_now <= 0:
            valid_previous = atrs[:i][atrs[:i] > 0]
            if len(valid_previous) == 0:
                skip_atr_fallback_count += 1
                continue
            fallback = np.median(valid_previous)
            if fallback <= 0 or np.isnan(fallback):
                skip_atr_fallback_count += 1
                continue
            atr_now = fallback

        # (B) Dynamic slippage
        slip_fixed = entry_price * SLIPPAGE_OFFSET
        slip_atr   = atr_now * SLIPPAGE_ATR_FRAC
        slip = max(slip_fixed, slip_atr)

        sl_price = entry_price - (atr_now * SL_MULTIPLIER) - slip
        tp_price = entry_price + (atr_now * TP_MULTIPLIER) + slip

        future_window = closes[i+1 : i+1+FUTURE_HORIZON]

        # (C) Constant future price?
        if np.all(future_window == future_window[0]):
            constant_price_count += 1
            continue

        # (D) TP vs SL first
        outcome = 0
        for price in future_window:
            if price >= tp_price:
                outcome = 1
                break
            elif price <= sl_price:
                outcome = 0
                break

        labels[i] = outcome
        labeled_count += 1

    # 3) Attach labels
    df["label"] = labels

    # 4) Drop the last FUTURE_HORIZON rows (lack future data)
    horizon_dropped_count = FUTURE_HORIZON
    drop_index_start = len(df) - FUTURE_HORIZON
    if drop_index_start < 0:
        horizon_dropped_count = 0
        drop_index_start = len(df)
    df = df.iloc[:drop_index_start].copy()
    df.reset_index(drop=True, inplace=True)

    # 5) Drop rows labeled -1 (either ATR fallback fail or constant future price)
    before_drop_minus1 = len(df)
    df = df[df["label"] != -1]
    df.reset_index(drop=True, inplace=True)
    dropped_minus1_count = before_drop_minus1 - len(df)

    # 6) Check final label distribution (0 vs. 1)
    label_0_count = (df["label"] == 0).sum()
    label_1_count = (df["label"] == 1).sum()
    total_after_label = len(df)

    print(f"Label distribution (after dropping -1): 0 => {label_0_count}, 1 => {label_1_count}")
    if total_after_label == 0:
        print("No labeled rows remain. Adjust your parameters or investigate data.")
        return

    # 7) Decide if undersampling is needed based on ratio
    #    If ratio of majority/minority > IMBALANCE_THRESHOLD, we do undersampling
    ratio = 0.0
    if label_0_count >= label_1_count and label_1_count != 0:
        ratio = label_0_count / float(label_1_count)
    elif label_1_count > label_0_count and label_0_count != 0:
        ratio = label_1_count / float(label_0_count)

    print(f"Current 0/1 ratio: ~{ratio:.2f} : 1 (majority : minority)")

    # We'll only undersample if ratio > IMBALANCE_THRESHOLD
    if ratio > IMBALANCE_THRESHOLD:
        print(f"Imbalance ratio {ratio:.2f} > {IMBALANCE_THRESHOLD}, triggering undersampling...")
        label_counts_before = Counter(df["label"])
        df_majority = df[df["label"] == (0 if label_0_count > label_1_count else 1)]
        df_minority = df[df["label"] == (1 if label_0_count > label_1_count else 0)]

        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=42
        )
        df = pd.concat([df_minority, df_majority_downsampled]).sample(frac=1, random_state=42)
        df.reset_index(drop=True, inplace=True)
        label_counts_after = Counter(df["label"])
        print(f"Label distribution after undersampling: {label_counts_after}")
    else:
        print(f"Ratio is within threshold ({IMBALANCE_THRESHOLD}), no undersampling applied.")

    # 8) Final logs and dataset size
    final_rows = len(df)
    total_dropped = initial_rows - final_rows
    pct_lost = total_dropped / initial_rows * 100.0

    print("=== LOG SUMMARY ===")
    print(f"Initial dataset rows: {initial_rows}")
    print(f"Skipped due to ATR fallback failure: {skip_atr_fallback_count}")
    print(f"Rows with constant future price: {constant_price_count}")
    print(f"Rows dropped from last FUTURE_HORIZON: {horizon_dropped_count}")
    print(f"Rows labeled -1 and dropped: {dropped_minus1_count}")
    print(f"Final dataset rows: {final_rows}")
    print(f"Total rows dropped: {total_dropped}")
    print(f"% of data lost: {pct_lost:.2f}%")

    # 9) Save final dataset
    df.to_csv(LABELED_DATA_PATH, index=False)
    print(f"[Stage 3] Labeled data saved to {LABELED_DATA_PATH}")

if __name__ == "__main__":
    main()
