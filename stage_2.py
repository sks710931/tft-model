import pandas as pd
import numpy as np
import ta  # pip install ta

CLEANED_DATA_PATH = "data/cleaned.csv"
ENGINEERED_DATA_PATH = "data/engineered.csv"

def add_technical_indicators(df):
    print("Starting feature addition ✅")
    print("Adding Moving Averages ✅")
    # ----- Moving Averages (Short/Medium) -----
    # Short window MAs
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_15"] = df["close"].rolling(window=15).mean()

    # Medium/longer window MAs
    df["ma_30"] = df["close"].rolling(window=30).mean()
    df["ma_60"] = df["close"].rolling(window=60).mean()

    print("Adding Moving Average Convergence Divergence ✅")
    # Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(
        close=df["close"], 
        window_fast=12, 
        window_slow=26, 
        window_sign=9
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    print("Adding Bollinger Bands ✅")
    # ----- Bollinger Bands -----
    bb = ta.volatility.BollingerBands(
        close=df["close"], 
        window=20, 
        window_dev=2
    )
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    # You can add %b or bandwidth if desired:
    df["bb_percent_b"] = bb.bollinger_pband()  # %b
    df["bb_bandwidth"] = bb.bollinger_wband()  # bandwidth

    print("Adding RSI - Relative Strength Index ✅")
    # ----- RSI (Relative Strength Index) -----
    df["rsi_14"] = ta.momentum.rsi(
        close=df["close"], 
        window=14
    )

    print("Adding Stochastic Oscillator ✅")
    # ----- Stochastic Oscillator (optional) -----
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], 
        low=df["low"], 
        close=df["close"], 
        window=14, 
        smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    print("Adding Average True Range ✅")
    # ----- ATR (Average True Range) for volatility -----
    df["atr_14"] = ta.volatility.average_true_range(
        high=df["high"], 
        low=df["low"], 
        close=df["close"], 
        window=14
    )

    print("Adding ADX - Average Directional Movement Index ✅")
    # ----- ADX (Average Directional Movement Index) for trend strength -----
    adx_indicator = ta.trend.ADXIndicator(
        high=df["high"], 
        low=df["low"], 
        close=df["close"], 
        window=14
    )
    df["adx"] = adx_indicator.adx()
    df["adx_pos"] = adx_indicator.adx_pos()  # +DI
    df["adx_neg"] = adx_indicator.adx_neg()  # -DI

    print("Adding ROC Rate of Change ✅")
    # ----- ROC (Rate of Change) -----
    df["roc_9"] = ta.momentum.roc(
        close=df["close"], 
        window=9
    )

    print("Adding Price Change from 1 bar ago ✅")
    # ----- Momentum: Price change from 1 bar ago -----
    df["price_diff_1"] = df["close"].diff(1)

    print("Adding Volume based features ✅")
    # ----- Volume-based features -----
    # 1) Rolling sum of volume (short window)
    df["vol_sum_5"] = df["volume"].rolling(window=5).sum()
    # 2) Volume ratio (current volume / average volume last 20 bars)
    df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(window=20).mean()
    
    print("Addition of Feature Indicators is done ✅")

    return df

def add_time_features(df):
    print("Starting to add time features ✅")
    print("Extracting hour, minute, day of week ✅")
    # Extract fields from timestamp
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    print("Adding Cyclical encoding for hour (24-hour cycle) ✅")
    # Cyclical encoding for hour (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    print("Adding Cyclical encoding for minute (60-min cycle) ✅")
    # Cyclical encoding for minute (60-min cycle)
    df["min_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["min_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    
    print("Adding Cyclical encoding for day (7-day cycle) ✅")
    # Cyclical encoding for day of week (7-day cycle)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    print("Time Features addition completed ✅")
    return df

def tag_market_regime(df):
    print("Starting to add Market Regime ✅")
    # We'll assume ADX is already computed in add_technical_indicators
    if "adx" not in df.columns:
        print("[Stage 2] WARNING: ADX not found. Did you add technical indicators first?")
        df["market_regime"] = 0
    else:
        df["market_regime"] = (df["adx"] > 25).astype(int)
    print("Completed adding Market Regime ✅")
    return df

def main():
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["timestamp"])
    print(f"Original dataset loaded: {df.shape[0]} rows")
    df["volume"] = df["volume"].replace(0, 1e-8)
    # Handle missing values
    filled_rows = df["close"].isna().sum()
    df.ffill(inplace=True)  
    df.dropna(inplace=True)
    dropped_rows = filled_rows + df["close"].isna().sum()

    print(f"Missing values handled: {filled_rows} rows forward-filled")
    print(f"Rows dropped after handling missing values: {dropped_rows}")
    
    # 1. Add technical indicators
    df = add_technical_indicators(df)
    
    # 2. Add time/cyclical features
    df = add_time_features(df)

    # 3. Tag simple market regime
    df = tag_market_regime(df)
    nan_counts = df.isna().sum().sort_values(ascending=False)
    print(nan_counts)

    # Drop NaN values introduced by rolling calculations
    initial_size = df.shape[0]
    df.dropna(inplace=True)
    final_size = df.shape[0]
    dropped_after_indicators = initial_size - final_size

    print(f"Rows dropped due to rolling indicators: {dropped_after_indicators}")
    print(f"Final dataset size after feature engineering: {df.shape[0]} rows")

    # 5. Save to new CSV
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(f"[Stage 2] Feature-engineered data saved to {ENGINEERED_DATA_PATH}")

if __name__ == "__main__":
    main()
