import pandas as pd
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
import os
import numpy as np

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
FUTURE_HORIZON = 15  # From Stage 3
ENTRY_FEE = 0.001    # 0.1%
EXIT_FEE = 0.001     # 0.1%
TOTAL_FEE = ENTRY_FEE + EXIT_FEE  # 0.2%

def evaluate_dataset(df):
    logger.info("=== Dataset Evaluation ===")
    total_rows = len(df)
    logger.info(f"Total rows: {total_rows}")
    
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    logger.info(f"Time range: {min_time} to {max_time}")
    
    # NaN Check
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.info("Columns with NaNs:")
        logger.info(nan_counts[nan_counts > 0].to_string())
    else:
        logger.info("No NaNs found.")
    
    # Label Distribution
    label_counts = Counter(df["label"])
    total_labels = sum(label_counts.values())
    logger.info(f"Label distribution: {label_counts}")
    logger.info(f"Short trades (0): {label_counts[0]} ({label_counts[0]/total_labels:.2%})")
    logger.info(f"Long trades (1): {label_counts[1]} ({label_counts[1]/total_labels:.2%})")
    logger.info(f"No trade (2): {label_counts[2]} ({label_counts[2]/total_labels:.2%})")
    
    # Temporal Coverage and Gaps
    df["year_month"] = df["timestamp"].dt.to_period("M")
    coverage = df["year_month"].value_counts().sort_index()
    logger.info("\n=== Coverage by Year-Month ===")
    for period, count in coverage.items():
        logger.info(f"{period}: {count} rows")
    
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
    gaps = df["time_diff"] > 180
    logger.info(f"\n=== Temporal Gaps > 3 Minutes ===")
    logger.info(f"Total gaps: {gaps.sum()}")
    if gaps.sum() > 0:
        logger.info(f"Example gaps: {df['timestamp'][gaps].head().to_list()}")
    
    # Profit Analysis
    logger.info("\n=== Profit Analysis (45-Min Horizon) ===")
    df["future_close"] = df["close"].shift(-FUTURE_HORIZON)
    df["profit"] = np.where(
        df["label"] == 1, (df["future_close"] / df["close"] - 1) - TOTAL_FEE,
        np.where(df["label"] == 0, (1 - df["future_close"] / df["close"]) - TOTAL_FEE, 0)
    )
    trade_profits = df[df["label"] != 2]["profit"]
    logger.info(f"Mean profit per trade: {float(trade_profits.mean()):.4f}")
    logger.info(f"Median profit: {float(trade_profits.median()):.4f}")
    logger.info(f"Min profit: {float(trade_profits.min()):.4f}")
    logger.info(f"Max profit: {float(trade_profits.max()):.4f}")
    profit_above_02 = (trade_profits >= 0.002).mean()
    logger.info(f"Trades ≥ 0.2% profit: {profit_above_02:.2%}")
    
    # Label Consistency
    logger.info("\n=== Label Consistency ===")
    consistent_longs = ((df["label"] == 1) & (df["profit"] > 0)).sum() / label_counts[1]
    consistent_shorts = ((df["label"] == 0) & (df["profit"] > 0)).sum() / label_counts[0]
    logger.info(f"Long trades (1) with positive profit: {consistent_longs:.2%}")
    logger.info(f"Short trades (0) with positive profit: {consistent_shorts:.2%}")
    
    # ATR Trend (Fixed to atr_3)
    logger.info("\n=== ATR_3 Trend by Year-Month ===")
    atr_trend = df.groupby("year_month")["atr_3"].mean()
    for period, atr in atr_trend.items():
        logger.info(f"{period}: Mean ATR_3 = {atr:.4f}")
    
    # Summary Statistics for Key Features
    summary_features = ["close", "atr_3", "ma_5", "rsi_14", "macd", "vol_change", "hl_spread"]
    logger.info("\n=== Summary Statistics ===")
    for feat in summary_features:
        if feat in df.columns:
            logger.info(f"Feature: {feat}")
            logger.info(df[feat].describe().to_string())
            logger.info("----------")
    
    # Clean up
    df.drop(columns=["year_month", "time_diff", "future_close", "profit"], inplace=True)

def main():
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    evaluate_dataset(df)

if __name__ == "__main__":
    main()