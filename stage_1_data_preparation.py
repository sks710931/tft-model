import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import os

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create rotating file handler
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "stage_1.log"),
    maxBytes=1_000_000,  # 1MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.handlers = []  # Clear any default handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
    logger.info("Missing values per column:")
    logger.info(missing_values.to_string())
    df.fillna(method="ffill", inplace=True)
    
    # Check for duplicates
    duplicate_count = df.duplicated(subset=["timestamp"]).sum()
    if duplicate_count > 0:
        logger.info(f"Found {duplicate_count} duplicate timestamp(s). Removing them.")
        df.drop_duplicates(subset=["timestamp"], inplace=True)
    
    # Ensure 3-minute sequence continuity and log gaps
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    full_range = pd.date_range(start=t_min, end=t_max, freq="3T")
    ref_df = pd.DataFrame({"timestamp": full_range})
    merged = ref_df.merge(df, on="timestamp", how="left")
    
    gaps = merged["timestamp"].diff().dt.total_seconds() > 180
    if gaps.sum() > 0:
        logger.info(f"Found {gaps.sum()} gaps > 3 minutes. First few: {merged['timestamp'][gaps].head().to_string()}")
    
    n_missing_rows = merged["open"].isnull().sum()
    if n_missing_rows > 0:
        logger.info(f"Found {n_missing_rows} missing row(s) in the 3-min sequence. Forward-filling.")
        merged.fillna(method="ffill", inplace=True)
    else:
        logger.info("No missing timestamps. Data is continuous.")
    
    # Save cleaned data
    merged.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Cleaned data saved to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()