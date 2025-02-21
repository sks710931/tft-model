import pandas as pd
import numpy as np
from collections import Counter

LABELED_DATA_PATH = "data/labeled_dynamic.csv"

def evaluate_dataset(df):
    print("=== Dataset Evaluation ===")
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    print(f"Time range: {min_time} to {max_time}")
    
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print("Columns with NaNs:")
        print(nan_counts[nan_counts > 0])
    else:
        print("No NaNs found.")
    
    label_counts = Counter(df["label"])
    print(f"Label distribution: {label_counts}")
    
    df["year_month"] = df["timestamp"].dt.to_period("M")
    coverage = df["year_month"].value_counts().sort_index()
    print("\n=== Coverage by Year-Month ===")
    for period, count in coverage.items():
        print(f"{period}: {count} rows")
    df.drop(columns="year_month", inplace=True)
    
    summary_features = ["close", "atr_14", "ma_60", "rsi_14"]
    print("\n=== Summary Statistics ===")
    for feat in summary_features:
        if feat in df.columns:
            print(f"Feature: {feat}")
            print(df[feat].describe())
            print("----------")

def main():
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    evaluate_dataset(df)

if __name__ == "__main__":
    main()