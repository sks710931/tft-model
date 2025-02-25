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
    logger.info("Starting technical indicator addition ✅")
    # Short-term MAs for HFT
    df["ma_3"] = df["close"].rolling(window=3, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    
    # Fast MACD for rapid signals
    macd = ta.trend.MACD(close=df["close"], window_fast=6, window_slow=13, window_sign=5)
    df["macd"] = macd.macd()
    df["macd_diff"] = macd.macd_diff()  # More sensitive than signal line
    
    # Short-term RSI
    df["rsi_7"] = ta.momentum.rsi(close=df["close"], window=7)
    
    # Volatility with shorter windows
    df["atr_3"] = ta.volatility.average_true_range(high=df["high"], low=df["low"], close=df["close"], window=3)
    df["atr_10"] = ta.volatility.average_true_range(high=df["high"], low=df["low"], close=df["close"], window=10)
    
    # Microstructural proxies
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]  # Normalized range
    df["oc_imbalance"] = (df["open"] - df["close"]).abs() / df["atr_3"]  # Candle body vs. volatility
    
    # Momentum and volume
    df["price_diff_1"] = df["close"].diff(1)
    df["price_diff_3"] = df["close"].diff(3)
    df["vol_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0)
    df["vol_spike"] = (df["volume"] > df["volume"].rolling(window=10, min_periods=1).mean() * 2).astype(int)
    
    logger.info("Technical indicator addition completed ✅")
    return df

def add_time_features(df):
    logger.info("Starting time feature addition ✅")
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    # Cyclic encodings (unchanged, still excellent for TFT)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["min_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["min_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    # Add weekend flag (crypto trades 24/7, but liquidity varies)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    logger.info("Time feature addition completed ✅")
    return df

def tag_market_regime(df):
    logger.info("Adding market regime ✅")
    # Multi-condition regime: trending (ADX), volatile (ATR), or calm
    df["regime_trend"] = (df["macd_diff"] > 0) & (df["atr_3"] > df["atr_3"].rolling(20).mean())
    df["regime_volatile"] = (df["atr_3"] > df["atr_3"].quantile(0.75)) & (df["vol_change"].abs() > 0.5)
    df["regime_calm"] = ~(df["regime_trend"] | df["regime_volatile"])
    df[["regime_trend", "regime_volatile", "regime_calm"]] = df[["regime_trend", "regime_volatile", "regime_calm"]].astype(int)
    logger.info("Market regimes (trend/volatile/calm) added ✅")
    return df

def main():
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["timestamp"])
    logger.info(f"Original dataset loaded: {df.shape[0]} rows")
    
    # Handle zero volume more naturally
    df["volume"] = df["volume"].replace(0, np.nan).ffill().fillna(1e-8)  # Preserve trend, avoid arbitrary small value
    df = df.ffill()  # Forward fill any remaining gaps
    
    # Add features
    df = add_technical_indicators(df)
    df = add_time_features(df)
    df = tag_market_regime(df)
    
    # Drop NaNs from rolling indicators (use min_periods to minimize loss)
    initial_size = df.shape[0]
    df = df.dropna()
    logger.info(f"Rows dropped due to rolling indicators: {initial_size - df.shape[0]}")
    logger.info(f"Final dataset size: {df.shape[0]} rows with {df.shape[1]} features")
    
    # Save
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    logger.info(f"Engineered data saved to {ENGINEERED_DATA_PATH}")

if __name__ == "__main__":
    main()