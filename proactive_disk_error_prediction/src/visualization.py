# src/visualization.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess
from config import DATA_PATH, LABEL_COLUMN

# Create results folder if not exists
SAVE_DIR = "../results/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_processed_data():
    """Loads and preprocesses dataset for visualization."""
    df = preprocess(DATA_PATH)
    return df


def correlation_heatmap(df):
    plt.figure(figsize=(22, 12))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")

    plt.savefig(f"{SAVE_DIR}/correlation_heatmap.png")
    plt.show()


def class_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[LABEL_COLUMN])
    plt.title("Failure Label Distribution")

    plt.savefig(f"{SAVE_DIR}/class_distribution.png")
    plt.show()


def histogram_plots(df):
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols].hist(figsize=(18, 18), bins=30)
    plt.tight_layout()

    plt.savefig(f"{SAVE_DIR}/histograms.png")
    plt.show()


def boxplots(df):
    numeric_cols = df.select_dtypes(include="number").columns

    plt.figure(figsize=(22, 10))
    sns.boxplot(data=df[numeric_cols])
    plt.xticks(rotation=90)
    plt.title("Boxplots for Numeric Features")

    plt.savefig(f"{SAVE_DIR}/boxplots.png")
    plt.show()


def pairplot(df):
    important = ["smart_5_raw", "smart_187_raw", "smart_197_raw", LABEL_COLUMN]
    subset = [col for col in important if col in df.columns]

    sns.pairplot(df[subset], hue=LABEL_COLUMN)
    plt.savefig(f"{SAVE_DIR}/pairplot.png")
    plt.show()


def run_all_visualizations():
    df = load_processed_data()
    print("Dataset loaded for visualization!")

    correlation_heatmap(df)
    class_distribution(df)
    histogram_plots(df)
    boxplots(df)
    pairplot(df)

    print(f"\nAll visualizations saved to: {SAVE_DIR}\n")


if __name__ == "__main__":
    run_all_visualizations()
