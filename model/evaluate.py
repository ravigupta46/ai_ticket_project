"""model/evaluate.py
Evaluate saved pipelines on held-out test split and write metrics.json.
"""
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model.config import METRICS_PATH


def evaluate_and_save_metrics(cat_pipe, pri_pipe, X_test, 
                              y_cat_test, y_pri_test,
                              cat_params=None, pri_params=None) -> dict:
    """Predict on X_test with both pipelines. Compute accuracy, 
    weighted precision, recall, f1 for each. Write METRICS_PATH 
    as JSON with keys: category_model, priority_model.
    Return the metrics dict."""
    # Category evaluation
    cat_preds = cat_pipe.predict(X_test)
    cat_metrics = {
        "accuracy": round(float(accuracy_score(y_cat_test, cat_preds)), 4),
        "precision": round(float(precision_score(y_cat_test, cat_preds, average='weighted', zero_division=0)), 4),
        "recall": round(float(recall_score(y_cat_test, cat_preds, average='weighted', zero_division=0)), 4),
        "f1": round(float(f1_score(y_cat_test, cat_preds, average='weighted', zero_division=0)), 4)
    }

    # Priority evaluation
    pri_preds = pri_pipe.predict(X_test)
    pri_metrics = {
        "accuracy": round(float(accuracy_score(y_pri_test, pri_preds)), 4),
        "precision": round(float(precision_score(y_pri_test, pri_preds, average='weighted', zero_division=0)), 4),
        "recall": round(float(recall_score(y_pri_test, pri_preds, average='weighted', zero_division=0)), 4),
        "f1": round(float(f1_score(y_pri_test, pri_preds, average='weighted', zero_division=0)), 4)
    }

    metrics_dict = {
        "dataset": "new_Customer_support_data.csv",
        "category_model": {
            "selected_model": type(cat_pipe.named_steps['classifier']).__name__,
            "best_params": {str(k): str(v) for k, v in (cat_params or {}).items()},
            "metrics": cat_metrics
        },
        "priority_model": {
            "selected_model": type(pri_pipe.named_steps['classifier']).__name__,
            "best_params": {str(k): str(v) for k, v in (pri_params or {}).items()},
            "metrics": pri_metrics
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"\nSaved Metrics to {METRICS_PATH}")
    print(f"Category Test Accuracy: {cat_metrics['accuracy']:.4f} | F1: {cat_metrics['f1']:.4f}")
    print(f"Priority Test Accuracy: {pri_metrics['accuracy']:.4f} | F1: {pri_metrics['f1']:.4f}")

    return metrics_dict
