"""model/train_category_model.py
Supervised customer support category classification pipeline using new_Customer_support_data.csv.
"""
import os
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ai_engine.config import ALLOWED_CATEGORIES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "new_Customer_support_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "category_pipeline.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
TRAINING_CSV_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")


def load_and_clean_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset, format subject and body, and validate category."""
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Synthesize text from subject and body
    subject = df['subject'].fillna('No Subject').astype(str).str.strip()
    body = df['body'].fillna('').astype(str).str.strip()
    text = (subject + " " + body).str.strip()

    # Determine category column (queue in new dataset, category in legacy)
    cat_col = 'queue' if 'queue' in df.columns else 'category'
    
    clean_df = pd.DataFrame({
        'text': text,
        'category': df[cat_col].astype(str).str.strip()
    })
    
    # Filter valid categories and minimum text length
    mask = (
        (clean_df['text'].str.len() > 5) &
        (clean_df['category'].isin(ALLOWED_CATEGORIES))
    )
    clean_df = clean_df[mask].drop_duplicates(subset=['text']).reset_index(drop=True)
    return clean_df


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("1. Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    df.to_csv(TRAINING_CSV_PATH, index=False, encoding='utf-8')
    print(f"Cleaned dataset: {len(df):,} records across {df['category'].nunique()} categories.")

    print("2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['category'],
        test_size=0.20,
        random_state=42,
        stratify=df['category']
    )

    print("3. Building and fitting TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 1))),
        ('classifier', LogisticRegression(C=2.0, max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    print("4. Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("5. Saving model and metrics...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    # Load existing metrics (to preserve priority_model if present)
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    metrics_data["category_model"] = {
        "selected_model": "Logistic Regression",
        "best_params": {'classifier__C': 2.0, 'tfidf__max_features': 5000, 'tfidf__ngram_range': (1, 1)},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train()
