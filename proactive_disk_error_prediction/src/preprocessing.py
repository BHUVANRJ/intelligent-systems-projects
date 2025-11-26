# src/preprocessing.py
import pandas as pd
import numpy as np

def load_dataset(path):
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    return df

def clean_dataset(df):
    """Basic cleaning: convert dates, drop unusable columns, fill missing values."""
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors="coerce")

    # Drop columns not needed
    drop_cols = ["serial_number"]
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    # Handle numeric columns
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Handle categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df

def encode_categorical(df):
    """Encode categorical columns using Label Encoding."""
    from sklearn.preprocessing import LabelEncoder
    label_cols = ['model']

    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df

def preprocess(path):
    df = load_dataset(path)
    df = clean_dataset(df)
    df = encode_categorical(df)
    return df
