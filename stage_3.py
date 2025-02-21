import pandas as pd
import numpy as np
from collections import Counter

ENGINEERED_DATA_PATH = "data/engineered.csv"
LABELED_DATA_PATH = "data/labeled_dynamic.csv"

FUTURE_HORIZON = 20   # 1 hour (20 * 3 minutes)
TP_PERCENT = 0.005    # 0.5% TP
SL_PERCENT = 0.002    # 0.2% SL
SLIPPAGE_OFFSET = 0.0005  # 0.05% slippage

def main():
    print("=== Stage 3: Labeling for HFT (0.5% TP, 0.2% SL, 1-Hour Horizon) ===")
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=["timestamp"])
    initial_rows = len(df)
    print(f"Initial dataset size: {initial_rows} rows")
    
    closes = df["close"].values
    labels = np.full(len(df), fill_value=2, dtype=int)  # Default to "no trade" (2)
    
    skip_count = 0
    long_count = 0
    short_count = 0
    
    # Label for long and short trades
    for i in range(len(df) - FUTURE_HORIZON):
        entry_price = closes[i]
        
        # Long trade: +0.5% TP, -0.2% SL
        long_tp_price = entry_price * (1 + TP_PERCENT) + (entry_price * SLIPPAGE_OFFSET)
        long_sl_price = entry_price * (1 - SL_PERCENT) - (entry_price * SLIPPAGE_OFFSET)
        
        # Short trade: -0.5% TP, +0.2% SL
        short_tp_price = entry_price * (1 - TP_PERCENT) - (entry_price * SLIPPAGE_OFFSET)
        short_sl_price = entry_price * (1 + SL_PERCENT) + (entry_price * SLIPPAGE_OFFSET)
        
        future_window = closes[i+1 : i+1+FUTURE_HORIZON]
        
        if np.all(future_window == future_window[0]):  # Constant price
            skip_count += 1
            continue
        
        # Check long trade outcome
        long_outcome = 2  # Default: no trade
        for price in future_window:
            if price >= long_tp_price:
                long_outcome = 1  # Long TP hit
                break
            elif price <= long_sl_price:
                long_outcome = 0  # Long SL hit (ignored for short)
                break
        
        # Check short trade outcome
        short_outcome = 2  # Default: no trade
        for price in future_window:
            if price <= short_tp_price:
                short_outcome = 0  # Short TP hit
                break
            elif price >= short_sl_price:
                short_outcome = 1  # Short SL hit (ignored for long)
                break
        
        # Assign label: prioritize tradable outcomes
        if long_outcome == 1:
            labels[i] = 1  # Long trade
            long_count += 1
        elif short_outcome == 0:
            labels[i] = 0  # Short trade
            short_count += 1
        # Else, remains 2 (no trade)
    
    df["label"] = labels
    df = df.iloc[:-FUTURE_HORIZON].copy()  # Drop rows without full horizon
    
    # Log label distribution
    label_counts = Counter(df["label"])
    print(f"Label distribution: {label_counts}")
    print(f"Short trades (0): {label_counts[0]}")
    print(f"Long trades (1): {label_counts[1]}")
    print(f"No trade (2): {label_counts[2]}")
    
    final_rows = len(df)
    print(f"Skipped constant price rows: {skip_count}")
    print(f"Long trade rows: {long_count}")
    print(f"Short trade rows: {short_count}")
    print(f"Final dataset rows: {final_rows}")
    print(f"% Data lost: {100 * (initial_rows - final_rows) / initial_rows:.2f}%")
    
    df.to_csv(LABELED_DATA_PATH, index=False)
    print(f"[Stage 3] Labeled data saved to {LABELED_DATA_PATH}")

if __name__ == "__main__":
    main()