import pandas as pd
import numpy as np
from collections import Counter

LABELED_DATA_PATH = "data/labeled_dynamic.csv"

def evaluate_dataset(df: pd.DataFrame):
    """
    Perform detailed metrics and logs about the labeled dataset.
    Modify or expand to suit your analysis needs.
    """

    print("=== Dataset Evaluation ===")

    # 1) Basic Info
    total_rows = len(df)
    print(f"Total rows in labeled dataset: {total_rows}")

    # 2) Time Range
    if "timestamp" in df.columns and not df["timestamp"].isnull().all():
        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()
        print(f"Data time range: {min_time} to {max_time}")
    else:
        print("No usable 'timestamp' column found or it's all NaN.")

    # 3) Check for leftover NaNs in features
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if not nan_cols.empty:
        print("Columns with NaNs (post-labeling):")
        for col, cnt in nan_cols.items():
            print(f"  {col}: {cnt} NaNs")
    else:
        print("No NaN values found in the dataset.")

    # 4) Label Distribution
    if "label" in df.columns:
        label_counts = Counter(df["label"])
        print(f"Label distribution: {label_counts}")
        # if it’s purely binary labeling
        num_label_0 = label_counts.get(0, 0)
        num_label_1 = label_counts.get(1, 0)
        if num_label_0 > 0 and num_label_1 > 0:
            ratio = max(num_label_0, num_label_1) / float(min(num_label_0, num_label_1))
            print(f"0/1 ratio: ~{ratio:.2f}:1")
        else:
            print("One label is zero or labeling is not strictly binary.")
    else:
        print("No 'label' column found in dataframe.")

    # 5) Data Coverage by Year or Month (Optional)
    #    If you want to see how data is distributed over time:
    if "timestamp" in df.columns:
        # Example grouping by year-month
        df["year_month"] = df["timestamp"].dt.to_period("M")
        coverage_counts = df["year_month"].value_counts().sort_index()
        print("\n=== Coverage by Year-Month ===")
        for period, count in coverage_counts.items():
            print(f"{period}: {count} rows")
        # drop helper column
        df.drop(columns="year_month", inplace=True, errors="ignore")

    # 6) Summary Stats for Key Features
    #    (e.g., checking the distribution of important indicators)
    summary_features = ["close", "atr_14", "ma_60", "rsi_14"]
    print("\n=== Summary Statistics for Key Features ===")
    for feat in summary_features:
        if feat in df.columns:
            desc = df[feat].describe()
            print(f"Feature: {feat}")
            print(desc)
            print("----------")
        else:
            print(f"Feature '{feat}' not found in dataset. Skipping.")
    
    print("=== End of Dataset Evaluation ===")


def main():
    # Load the labeled dataset
    df = pd.read_csv(LABELED_DATA_PATH, parse_dates=["timestamp"])
    evaluate_dataset(df)

if __name__ == "__main__":
    main()
