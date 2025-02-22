# stage_1.py
import pandas as pd

RAW_DATA_PATH = "data/data.csv"
CLEANED_DATA_PATH = "data/cleaned.csv"

def main():
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Sort by timestamp
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("[Stage 1] Missing values per column:")
    print(missing_values)
    df.fillna(method="ffill", inplace=True)  # Forward-fill missing OHLCV values
    
    # Check for duplicates
    duplicate_count = df.duplicated(subset=["timestamp"]).sum()
    if duplicate_count > 0:
        print(f"[Stage 1] Found {duplicate_count} duplicate timestamp(s). Removing them.")
        df.drop_duplicates(subset=["timestamp"], inplace=True)
    
    # Ensure 3-minute sequence continuity
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    full_range = pd.date_range(start=t_min, end=t_max, freq="3T")  # 3-minute intervals
    ref_df = pd.DataFrame({"timestamp": full_range})
    merged = ref_df.merge(df, on="timestamp", how="left")
    
    n_missing_rows = merged["open"].isnull().sum()
    if n_missing_rows > 0:
        print(f"[Stage 1] Found {n_missing_rows} missing row(s) in the 3-min sequence. Forward-filling.")
        merged.fillna(method="ffill", inplace=True)
    else:
        print("[Stage 1] No missing timestamps. Data is continuous.")
    
    # Save cleaned data
    merged.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[Stage 1] Cleaned data saved to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()