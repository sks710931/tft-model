import pandas as pd
import numpy as np
import ta
import logging
from logging.handlers import RotatingFileHandler
import os

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, "stage_2.log"),
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

CLEANED_DATA_PATH = "data/cleaned.csv"
ENGINEERED_DATA_PATH = "data/engineered.csv"

def add_technical_indicators(df):
    logger.info("Starting feature addition ✅")
    df["ma_1"] = df["close"].rolling(window=1).mean()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_15"] = df["close"].rolling(window=15).mean()
    df["ma_30"] = df["close"].rolling(window=30).mean()
    
    macd = ta.trend.MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    
    df["rsi_14"] = ta.momentum.rsi(close=df["close"], window=14)
    
    df["atr_5"] = ta.volatility.average_true_range(high=df["high"], low=df["low"], close=df["close"], window=5)
    df["atr_14"] = ta.volatility.average_true_range(high=df["high"], low=df["low"], close=df["close"], window=14)
    
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    
    df["price_diff_1"] = df["close"].diff(1)
    df["price_diff_5"] = df["close"].diff(5)
    
    df["vol_sum_5"] = df["volume"].rolling(window=5).sum()
    df["vol_ratio_5"] = df["volume"] / df["volume"].rolling(window=5).mean()
    
    df["volatility_ratio"] = df["atr_5"] / df["atr_14"]
    
    logger.info("Feature addition completed ✅")
    return df

def add_time_features(df):
    logger.info("Starting time feature addition ✅")
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["min_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["min_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    logger.info("Time feature addition completed ✅")
    return df

def tag_market_regime(df):
    logger.info("Adding market regime ✅")
    df["market_regime"] = (df["adx"] > 25).astype(int)
    logger.info("Market regime added ✅")
    return df

def main():
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["timestamp"])
    logger.info(f"Original dataset loaded: {df.shape[0]} rows")
    df["volume"] = df["volume"].replace(0, 1e-8)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    df = add_technical_indicators(df)
    df = add_time_features(df)
    df = tag_market_regime(df)
    
    initial_size = df.shape[0]
    df.dropna(inplace=True)
    logger.info(f"Rows dropped due to rolling indicators: {initial_size - df.shape[0]}")
    logger.info(f"Final dataset size: {df.shape[0]} rows")
    
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    logger.info(f"Engineered data saved to {ENGINEERED_DATA_PATH}")

if __name__ == "__main__":
    main()