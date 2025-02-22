# stage_3.py
import pandas as pd
import numpy as np
from collections import Counter

# File paths
ENGINEERED_DATA_PATH = "data/engineered.csv"
LABELED_DATA_PATH = "data/labeled_dynamic.csv"

# Configuration constants
FUTURE_HORIZON = 20   # 1 hour (20 * 3-minute bars)
ENTRY_FEE = 0.001     # 0.1% entry fee
EXIT_FEE = 0.001      # 0.1% exit fee
TOTAL_FEE = ENTRY_FEE + EXIT_FEE  # 0.2% total fee
PROFIT_MARGIN = 0.002  # 0.2% desired profit margin (reduced from 0.3%)
MOVEMENT_THRESHOLD = TOTAL_FEE + PROFIT_MARGIN  # 0.4% total threshold

def main():
    print("=== Stage 3: Labeling for HFT (Fee-Aware Direction-Based, 1-Hour Horizon) ===")
    
    # Load the engineered dataset
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=["timestamp"])
    initial_rows = len(df)
    print(f"Initial dataset size: {initial_rows} rows")
    
    # Extract closing prices and initialize labels (default: no trade)
    closes = df["close"].values
    labels = np.full(len(df), fill_value=2, dtype=int)  # 2 = no trade
    
    # Iterate through the dataset (excluding the last FUTURE_HORIZON rows)
    for i in range(len(df) - FUTURE_HORIZON):
        entry_price = closes[i]
        future_window = closes[i+1 : i+1+FUTURE_HORIZON]
        
        if len(future_window) == 0:
            continue
        
        # Check for first breach of threshold
        for price in future_window:
            # Calculate percentage movements
            up_movement = (price / entry_price - 1) - TOTAL_FEE
            down_movement = (1 - price / entry_price) - TOTAL_FEE
            
            # Assign label based on first direction to meet threshold
            if up_movement >= PROFIT_MARGIN:
                labels[i] = 1  # Long trade
                break
            elif down_movement >= PROFIT_MARGIN:
                labels[i] = 0  # Short trade
                break
        else:
            labels[i] = 2  # No trade if neither threshold is met
    
    # Add labels to the dataframe and trim rows without a full horizon
    df["label"] = labels
    df = df.iloc[:-FUTURE_HORIZON].copy()
    
    # Log label distribution
    label_counts = Counter(df["label"])
    print(f"Label distribution: {label_counts}")
    print(f"Short trades (0): {label_counts[0]}")
    print(f"Long trades (1): {label_counts[1]}")
    print(f"No trade (2): {label_counts[2]}")
    
    # Log final dataset size
    final_rows = len(df)
    print(f"Final dataset rows: {final_rows}")
    print(f"% Data lost: {100 * (initial_rows - final_rows) / initial_rows:.2f}%")
    
    # Save the labeled dataset
    df.to_csv(LABELED_DATA_PATH, index=False)
    print(f"[Stage 3] Labeled data saved to {LABELED_DATA_PATH}")

if __name__ == "__main__":
    main()