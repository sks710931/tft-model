import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/data.csv"
CLEANED_DATA_PATH = "data/cleaned.csv"

def main():
    # 1. Load raw data
    #    - parse timestamp from milliseconds -> datetime
    df = pd.read_csv(RAW_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # 2. Sort by timestamp
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 3. Basic checks for missing or duplicate rows
    #    - Check for NaNs in any column
    missing_values = df.isnull().sum()
    print("[Stage 1] Missing values per column:")
    print(missing_values)
    
    #    - Check for duplicate timestamps
    duplicate_count = df.duplicated(subset=["timestamp"]).sum()
    if duplicate_count > 0:
        print(f"[Stage 1] Found {duplicate_count} duplicate timestamp(s). Removing them.")
        df.drop_duplicates(subset=["timestamp"], inplace=True)
    
    # 4. Check for missing timestamps in the 3-min sequence
    #    - Create a reference range from the min to max timestamp with 3-min frequency
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    full_range = pd.date_range(start=t_min, end=t_max, freq="3T")  # 3-minute intervals
    
    # Merge df onto this reference to see if any timestamps are missing
    ref_df = pd.DataFrame({"timestamp": full_range})
    merged = ref_df.merge(df, on="timestamp", how="left")
    
    # Check how many newly introduced NaNs appear
    # (In theory, if your data is truly complete, this should be 0)
    n_missing_rows = merged["open"].isnull().sum()  # or check any column
    if n_missing_rows > 0:
        print(f"[Stage 1] WARNING: Found {n_missing_rows} missing row(s) in the 3-min sequence.")
        # If you want to fill them, you could do:
        # merged.fillna(method="ffill", inplace=True)
        # or drop them, etc.
    else:
        print("[Stage 1] No missing timestamps. Data is continuous at 3-min intervals.")
    
    # 5. For final dataset, we either keep the merged version or, if no misses,
    #    we can just keep the original df. Below we keep the merged to ensure
    #    every 3-minute slot is represented (if you want forward-fill).
    #    If you truly have no missing data, merged ~ df except for forward-filled rows.
    
    # Option A: if no missing => just use df
    # final_df = df
    
    # Option B: if missing => keep merged and fill:
    # merged.fillna(method="ffill", inplace=True)
    # final_df = merged
    
    # For demonstration, let's keep "merged" in case you want that full range
    merged.reset_index(drop=True, inplace=True)
    
    # 6. Save cleaned file
    merged.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[Stage 1] Cleaned data (with reference timestamps) saved to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()
