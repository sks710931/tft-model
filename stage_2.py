# stage_2.py
import pandas as pd
import numpy as np
import ta

CLEANED_DATA_PATH = "data/cleaned.csv"
ENGINEERED_DATA_PATH = "data/engineered.csv"

def add_technical_indicators(df):
    print("Starting feature addition ✅")
    # Moving Averages
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_15"] = df["close"].rolling(window=15).mean()
    df["ma_30"] = df["close"].rolling(window=30).mean()
    df["ma_60"] = df["close"].rolling(window=60).mean()
    df["ma_120"] = df["close"].rolling(window=120).mean()  # 6-hour MA
    
    # MACD
    macd = ta.trend.MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    
    # RSI
    df["rsi_14"] = ta.momentum.rsi(close=df["close"], window=14)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    
    # ATR (for labeling and features)
    df["atr_14"] = ta.volatility.average_true_range(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr_60"] = ta.volatility.average_true_range(high=df["high"], low=df["low"], close=df["close"], window=60)
    
    # ADX
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    
    # ROC
    df["roc_9"] = ta.momentum.roc(close=df["close"], window=9)
    
    # Price Momentum
    df["price_diff_1"] = df["close"].diff(1)
    df["price_diff_60"] = df["close"].diff(60)  # 3-hour lag
    
    # Volume Features
    df["vol_sum_5"] = df["volume"].rolling(window=5).sum()
    df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(window=20).mean()
    
    # Volatility Spike
    df["volatility_spike"] = (df["atr_14"] / df["atr_14"].rolling(window=60).mean()) > 1.5
    
    # Additional Indicators
    cci = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20)
    df["cci_20"] = cci.cci()

    willr = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=14)
    df["willr_14"] = willr.williams_r()

    df["rsi_14_lag5"] = df["rsi_14"].shift(5)  # RSI from 15 minutes ago
    
    # Normalize numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    print("Feature addition completed ✅")
    return df

def add_time_features(df):
    print("Starting time feature addition ✅")
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["min_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["min_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    print("Time feature addition completed ✅")
    return df

def tag_market_regime(df):
    print("Adding market regime ✅")
    df["market_regime"] = (df["adx"] > 25).astype(int)
    print("Market regime added ✅")
    return df

def main():
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["timestamp"])
    print(f"Original dataset loaded: {df.shape[0]} rows")
    df["volume"] = df["volume"].replace(0, 1e-8)  # Avoid division by zero
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    df = add_technical_indicators(df)
    df = add_time_features(df)
    df = tag_market_regime(df)
    
    initial_size = df.shape[0]
    df.dropna(inplace=True)
    print(f"Rows dropped due to rolling indicators: {initial_size - df.shape[0]}")
    print(f"Final dataset size: {df.shape[0]} rows")
    
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(f"[Stage 2] Engineered data saved to {ENGINEERED_DATA_PATH}")

if __name__ == "__main__":
    main()