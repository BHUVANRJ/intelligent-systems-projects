# src/train_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess
from feature_engineering import add_features, split_features_labels
from utils import evaluate_model

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

def train_all_models(data_path):
    # Load + preprocess
    df = preprocess(data_path)
    df = add_features(df)

    # Split features
    X, y = split_features_labels(df)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=500),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
        ),
        "SVM": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LightGBM": LGBMClassifier(),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500)
    }

    # Train all models
    for name, model in models.items():
        print(f"\n🔹 Training: {name}")
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    train_all_models("C:\\UMICH\\intelligent-systems-projects\\proactive_disk_error_prediction\\data\\final_processed_smart_data_for_ml_15days for normalize.csv")
