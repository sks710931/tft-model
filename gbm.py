import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss
import os

# Configuration
INPUT_DIR = "data/walk_forward_splits_gap"
OUTPUT_RESULTS = "data/lightgbm_results.csv"
SEQ_LENGTH = 20  # 1-hour horizon (20 bars)

# Feature engineering: Flatten past 20 bars
def create_lagged_features(df, seq_length):
    features = []
    for col in df.drop(columns=["timestamp", "label"]).columns:
        for lag in range(seq_length):
            features.append(df[col].shift(lag).rename(f"{col}_lag{lag}"))
    return pd.concat(features + [df["label"]], axis=1).dropna()

# Load and prepare Stage 5 data
def load_and_prepare_data(file_prefix, window_index):
    file_path = os.path.join(INPUT_DIR, f"{file_prefix}_{window_index}.csv")
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    return create_lagged_features(df, SEQ_LENGTH)

# Training and evaluation function
def train_and_evaluate(train_df, val_df, test_df, params, window_index):
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train, weight=[2.0 if y != 2 else 0.5 for y in y_train])
    val_data = lgb.Dataset(X_val, label=y_val, weight=[2.0 if y != 2 else 0.5 for y in y_val])

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )

    # Validation loss
    y_pred_proba_val = model.predict(X_val)
    val_loss = log_loss(y_val, y_pred_proba_val)

    # Test evaluation
    y_pred_proba_test = model.predict(X_test)
    y_pred_test = np.argmax(y_pred_proba_test, axis=1)
    
    # PnL simulation (adapted from TFT)
    closes = test_df["close"].values[-len(y_pred_test):]
    gains = []
    num_trades = 0
    for i, pred in enumerate(y_pred_test):
        if i >= len(closes) - 1 or pred == 2:
            gains.append(0.0)
            continue
        entry_price = closes[i]
        if pred == 1:  # Long
            tp_price = entry_price * 1.005
            sl_price = entry_price * 0.998
            future_prices = closes[i+1 : min(i+21, len(closes))]
            hit_tp = any(p >= tp_price for p in future_prices)
            hit_sl = any(p <= sl_price for p in future_prices)
            if hit_tp:
                gain = 0.005 - 0.002  # TP - 2 * fee
                gains.append(gain)
            elif hit_sl:
                gain = -0.002 - 0.002  # SL - 2 * fee
                gains.append(gain)
            else:
                gains.append(-0.002)  # Fee only
            num_trades += 1
        elif pred == 0:  # Short
            tp_price = entry_price * 0.995
            sl_price = entry_price * 1.002
            future_prices = closes[i+1 : min(i+21, len(closes))]
            hit_tp = any(p <= tp_price for p in future_prices)
            hit_sl = any(p >= sl_price for p in future_prices)
            if hit_tp:
                gain = 0.005 - 0.002
                gains.append(gain)
            elif hit_sl:
                gain = -0.002 - 0.002
                gains.append(gain)
            else:
                gains.append(-0.002)
            num_trades += 1
    
    total_pnl = sum(gains)
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0

    return {
        "val_loss": val_loss,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "avg_pnl_per_trade": avg_pnl,
    }

# Main execution
def main():
    print("=== LightGBM Training for HFT (0.5% TP, 0.2% SL, 1-Hour Horizon) ===")
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    results = []
    all_files = os.listdir(INPUT_DIR)
    pattern = re.compile(r"train_(\d+)\.csv")
    train_files = sorted([f for f in all_files if pattern.match(f)], key=lambda x: int(pattern.match(x).group(1)))

    for idx, train_file in enumerate(train_files):
        window_index = int(pattern.match(train_file).group(1))
        val_file = f"val_{window_index}.csv"
        test_file = f"test_{window_index}.csv"
        path_train = os.path.join(INPUT_DIR, train_file)
        path_val = os.path.join(INPUT_DIR, val_file)
        path_test = os.path.join(INPUT_DIR, test_file)

        if not (os.path.exists(path_val) and os.path.exists(path_test)):
            print(f"Window {window_index}: missing val/test file, skipping.")
            continue

        train_df = pd.read_csv(path_train, parse_dates=["timestamp"])
        val_df = pd.read_csv(path_val, parse_dates=["timestamp"])
        test_df = pd.read_csv(path_test, parse_dates=["timestamp"])

        if len(train_df) < MIN_TRAIN_SIZE or len(val_df) < MIN_VAL_SIZE or len(test_df) < MIN_TEST_SIZE:
            print(f"Skipping window {window_index}: insufficient data.")
            continue

        print(f"\n=== Window {window_index} ===")
        train_data = create_lagged_features(train_df, SEQ_LENGTH)
        val_data = create_lagged_features(val_df, SEQ_LENGTH)
        test_data = create_lagged_features(test_df, SEQ_LENGTH)

        metrics = train_and_evaluate(train_data, val_data, test_data, params, window_index)
        print(f"Window {window_index} => val_loss={metrics['val_loss']:.4f}, metrics={metrics}")

        row = {
            "window_index": window_index,
            "train_start": train_df["timestamp"].min(),
            "train_end": train_df["timestamp"].max(),
            "val_start": val_df["timestamp"].min(),
            "val_end": val_df["timestamp"].max(),
            "test_start": test_df["timestamp"].min(),
            "test_end": test_df["timestamp"].max(),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "val_loss": metrics["val_loss"],
            "total_pnl": metrics["total_pnl"],
            "num_trades": metrics["num_trades"],
            "avg_pnl_per_trade": metrics["avg_pnl_per_trade"],
        }
        results.append(row)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_RESULTS, index=False)
    print(f"\nResults saved to {OUTPUT_RESULTS}")
    print(df_results)

if __name__ == "__main__":
    main()