# src/feature_engineering.py
import pandas as pd

def add_features(df):
    """Add rolling statistics and engineered features."""
    # Example SMART attributes (modify if needed)
    smart_cols = ["smart_5_raw", "smart_187_raw", "smart_197_raw"]

    for col in smart_cols:
        if col in df.columns:
            df[f"{col}_roll_mean_10"] = df[col].rolling(10).mean()
            df[f"{col}_roll_std_5"]  = df[col].rolling(5).std()

    df = df.fillna(0)
    return df

def split_features_labels(df, label="will_fail_in_15_days"):
    """Split X and y."""
    X = df.drop([label], axis=1)
    y = df[label]
    return X, y
