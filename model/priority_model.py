"""model/priority_model.py
Priority classification pipeline trained on new_Customer_support_data.csv.
"""
import os
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ai_engine.rules import rule_based_priority

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'new_Customer_support_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'output')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.json')


def load_priority_dataset(custom_path=None):
    """
    Load priority dataset. Supports new_Customer_support_data.csv directly with verified labels.
    """
    path = custom_path or os.environ.get("PRIORITY_DATA_PATH") or DATA_PATH
    if not os.path.exists(path):
        legacy_path = os.path.join(BASE_DIR, 'data', 'Customer_support_data.csv')
        path = legacy_path if os.path.exists(legacy_path) else path

    print(f"[LOAD] Loading priority data from: {path}")
    df = pd.read_csv(path, encoding='latin1' if 'new_' in path else 'utf-8', low_memory=False)

    if 'subject' in df.columns and 'body' in df.columns:
        subject = df['subject'].fillna('No Subject').astype(str).str.strip()
        body = df['body'].fillna('').astype(str).str.strip()
        df['text'] = (subject + " " + body).str.strip()
    elif 'Customer Remarks' in df.columns:
        df['text'] = df['Customer Remarks'].astype(str).str.strip()

    if 'priority' in df.columns:
        df['priority'] = df['priority'].astype(str).str.strip().str.capitalize()
        df['is_human_verified'] = True
    else:
        df['priority'] = df['text'].apply(rule_based_priority)
        df['is_human_verified'] = False

    df = df[df['text'].str.len() > 5].drop_duplicates(subset=['text']).reset_index(drop=True)
    return df


def train_priority_pipeline():
    df = load_priority_dataset()
    print(f"[DATA] Total usable records: {len(df):,}")
    print("[DISTRIBUTION] Priority class counts:")
    print(df['priority'].value_counts())

    X = df['text']
    y = df['priority']

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 1))),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=35, random_state=42, n_jobs=-1))
    ])

    print("\n[TRAINING] Fitting Random Forest Priority Pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    print(f"[METRICS] Priority Accuracy: {acc*100:.2f}% | Weighted F1: {f1:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "priority_pipeline.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    # Save to metrics.json
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    metrics_data["priority_model"] = {
        "selected_model": "Random Forest",
        "best_params": {'classifier__max_depth': 35, 'classifier__n_estimators': 150, 'tfidf__max_features': 3000, 'tfidf__ngram_range': (1, 1)},
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"[SAVED] priority_pipeline.pkl and metrics.json updated.")
    return pipeline


if __name__ == "__main__":
    train_priority_pipeline()
