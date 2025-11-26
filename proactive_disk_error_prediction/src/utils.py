# src/utils.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

def evaluate_model(model, X_test, y_test):
    """Print accuracy, precision, recall, F1, AUC."""
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:,1]
        print("ROC-AUC:", roc_auc_score(y_test, probas))

    print("----------------------")
