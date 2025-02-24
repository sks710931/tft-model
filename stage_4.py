import pandas as pd
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
import os

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, "stage_4.log"),
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

def evaluate_dataset(df):
    logger.info("=== Dataset Evaluation ===")
    total_rows = len(df)
    logger.info(f"Total rows: {total_rows}")
    
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    logger.info(f"Time range: {min_time} to {max_time}")
    
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.info("Columns with NaNs:")
        logger.info(nan_counts[nan_counts > 0].to_string())
    else:
        logger.info("No NaNs found.")
    
    label_counts = Counter(df["label"])
    logger.info(f"Label distribution: {label_counts}")
    
    df["year_month"] = df["timestamp"].dt.to_period("M")
    coverage = df["year_month"].value_counts().sort_index()
    logger.info("\n=== Coverage by Year-Month ===")
    for period, count in coverage.items():
        logger.info(f"{period}: {count} rows")
    
    logger.info("\n=== ATR_5 Trend by Year-Month ===")
    atr_trend = df.groupby("year_month")["atr_5"].mean()
    for period, atr in atr_trend.items():
        logger.info(f"{period}: Mean ATR_5 = {atr:.4f}")
    
    df.drop(columns="year_month", inplace=True)
    
    summary_features = ["close", "atr_5", "ma_5", "rsi_14"]
    logger.info("\n=== Summary Statistics ===")
    for feat in summary_features:
        if feat in df.columns:
            logger.info(f"Feature: {feat}")
            logger.info(df[feat].describe().to_string())
            logger.info("----------")

def main():
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    evaluate_dataset(df)

if __name__ == "__main__":
    main()