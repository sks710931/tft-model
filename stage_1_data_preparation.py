import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import os
import numpy as np
from scipy import stats

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, "stage_1.log"),
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

RAW_DATA_PATH = "data/data.csv"
CLEANED_DATA_PATH = "data/cleaned.csv"

def validate_data(df):
    """Validate OHLCV data integrity."""
    errors = []
    for col in ["open", "high", "low", "close", "volume"]:
        if df[col].lt(0).any():
            errors.append(f"{col} has negative values")
    if (df["high"] < df["low"]).any():
        errors.append("High < Low detected")
    if errors:
        logger.warning("Data validation issues: " + "; ".join(errors))
    else:
        logger.info("Data validation passed.")

def detect_outliers(df, column, z_threshold=6):
    """Detect and log outliers using Z-score."""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[z_scores > z_threshold]
    if not outliers.empty:
        logger.info(f"Found {len(outliers)} outliers in {column}: {outliers['timestamp'].head().to_list()}")
    return outliers.index

def main():
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    logger.info(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Initial checks
    missing_values = df.isnull().sum()
    logger.info("Missing values per column:\n" + missing_values.to_string())
    df = df.fillna(method="ffill").fillna(method="bfill")  # Forward then backward fill for edges
    
    duplicate_count = df.duplicated(subset=["timestamp"]).sum()
    if duplicate_count > 0:
        logger.info(f"Found {duplicate_count} duplicate timestamps. Keeping last.")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
    
    # Validate data
    validate_data(df)
    
    # Outlier detection (optional capping)
    for col in ["open", "high", "low", "close"]:
        outlier_idx = detect_outliers(df, col)
        # Optionally cap outliers: df.loc[outlier_idx, col] = df[col].quantile(0.99)
    
    # Ensure 3-minute continuity
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    full_range = pd.date_range(start=t_min, end=t_max, freq="3T")
    ref_df = pd.DataFrame({"timestamp": full_range})
    merged = ref_df.merge(df, on="timestamp", how="left")
    
    # Analyze gaps
    gaps = merged["timestamp"].diff().dt.total_seconds() > 180
    gap_sizes = merged["timestamp"].diff().dt.total_seconds()[gaps].dropna() / 60  # In minutes
    if gaps.sum() > 0:
        logger.info(f"Found {gaps.sum()} gaps > 3 minutes. Median gap size: {gap_sizes.median():.2f} minutes")
        logger.info(f"Gap size distribution: {gap_sizes.describe().to_string()}")
    
    # Smart gap filling
    n_missing = merged["open"].isnull().sum()
    if n_missing > 0:
        gap_mask = merged["open"].isnull()
        large_gaps = gap_mask.rolling(window=10, min_periods=1).sum() > 5  # Flag gaps > 15 min
        if large_gaps.sum() > 0:
            logger.warning(f"Detected {large_gaps.sum()} large gaps (>15 min). Consider interpolation.")
            # Linear interpolation for large gaps
            merged.loc[large_gaps, ["open", "high", "low", "close", "volume"]] = np.nan
            merged = merged.interpolate(method="linear")
        merged = merged.fillna(method="ffill").fillna(method="bfill")
        logger.info(f"Filled {n_missing} missing rows in 3-min sequence.")
    else:
        logger.info("No missing timestamps. Data is continuous.")
    
    # Add basic time features for TFT
    merged["hour"] = merged["timestamp"].dt.hour
    merged["day_of_week"] = merged["timestamp"].dt.dayofweek
    merged["month"] = merged["timestamp"].dt.month
    
    # Save cleaned data
    merged.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Cleaned data with {len(merged)} rows saved to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()