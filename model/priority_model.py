"""model/priority_model.py
Priority classifier training with weak-supervision transparency.

NOTE ON WEAK LABELS:
In the absence of human agent-labeled priority annotations, training labels are generated
heuristically using the shared priority rules from `ai_engine.config`. The Random Forest
model therefore learns a smoothed representation of the rule keyword triggers (weak supervision).
This model is maintained for architecture compatibility, but evaluation scores reflect
agreement with heuristic rules rather than ground-truth human judgments.
"""
import os
import sys
import json
import pickle
from datetime import datetime, timezone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_engine.rules import rule_based_priority

OUTPUT_DIR = os.path.join(BASE_DIR, 'model', 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_PATH = os.path.join(OUTPUT_DIR, 'training_data.csv')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.json')


def load_priority_dataset(custom_path: str = None) -> pd.DataFrame:
    """Documented hook for loading real agent-labeled priority data when available.

    To use real ground-truth data:
    1. Provide a CSV containing 'text' and 'priority' (High/Medium/Low) columns.
    2. Pass the path to this function or set the PRIORITY_DATA_PATH environment variable.
    3. If no verified path is given, it gracefully falls back to weak labeling from customer data.
    """
    path = custom_path or os.environ.get("PRIORITY_DATA_PATH")
    if path and os.path.exists(path):
        print(f"[DATA HOOK] Loading verified agent-labeled priority data from: {path}")
        df = pd.read_csv(path)
        if 'text' in df.columns and 'priority' in df.columns:
            df['is_human_verified'] = True
            return df.dropna(subset=['text', 'priority']).reset_index(drop=True)

    # Heuristic weak-label fallback
    src_csv = DATA_PATH if os.path.exists(DATA_PATH) else os.path.join(BASE_DIR, 'data', 'Customer_support_data.csv')
    print(f"[LOAD] Weak-supervision: generating labels from: {src_csv}")
    df = pd.read_csv(src_csv, low_memory=False)
    text_col = 'text' if 'text' in df.columns else 'Customer Remarks'

    df = df[df[text_col].notnull()].copy()
    df['text'] = df[text_col].astype(str).str.strip()
    df = df[df['text'].str.len() > 8].drop_duplicates(subset=['text']).reset_index(drop=True)

    # Generate labels using the unified shared keyword rules (no duplicated keyword lists)
    df['priority'] = df['text'].apply(rule_based_priority)
    df['is_human_verified'] = False
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
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2)),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1))
    ])

    print("\n[TRAINING] Fitting Random Forest Priority Pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    print(f"[METRICS] Priority Accuracy: {acc*100:.2f}% | Weighted F1: {f1:.4f}")
    is_verified = bool(df['is_human_verified'].iloc[0]) if 'is_human_verified' in df.columns else False
    note = "human-verified" if is_verified else "weak-label (rule-derived), not human-verified"

    # Save Pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, 'priority_pipeline.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)

    # Backwards compatibility artifacts
    with open(os.path.join(MODEL_DIR, 'priority_model.pkl'), 'wb') as f:
        pickle.dump(pipeline.named_steps['clf'], f)
    with open(os.path.join(MODEL_DIR, 'priority_vectorizer.pkl'), 'wb') as f:
        pickle.dump(pipeline.named_steps['tfidf'], f)
    with open(os.path.join(OUTPUT_DIR, 'priority_accuracy.txt'), 'w') as f:
        f.write(str(round(acc, 4)))

    # Update metrics.json
    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    metrics['priority_model'] = {
        "model_type": "RandomForestClassifier",
        "evaluation_note": note,
        "accuracy": round(acc, 4),
        "weighted_f1": round(f1, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"[OK] Saved priority_pipeline.pkl and updated metrics.json with '{note}'.")


if __name__ == '__main__':
    train_priority_pipeline()