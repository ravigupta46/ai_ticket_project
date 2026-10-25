"""model/train_category_model.py
Simple supervised customer support category classification pipeline.

Flow:
1. Load dataset
2. Clean text, drop duplicates, filter valid categories
3. Stratified train/test split
4. Build TF-IDF + Logistic Regression pipeline
5. Fit pipeline on training set
6. Predict on test set
7. Evaluate metrics (accuracy, precision, recall, f1)
8. Save model pipeline and metrics
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Customer_support_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "category_pipeline.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
TRAINING_CSV_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")

ALLOWED_CATEGORIES = [
    'Product Queries', 'Returns', 'Order Related', 'Refund Related',
    'Payments related', 'Cancellation', 'Shopzilla Related', 'Feedback',
    'Offers & Cashback', 'App/website', 'Onboarding related', 'Others'
]


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load dataset, filter non-null valid remarks and valid categories, then deduplicate."""
    df = pd.read_csv(file_path, low_memory=False)
    mask = (
        df['Customer Remarks'].notnull() &
        (df['Customer Remarks'].astype(str).str.strip().str.len() > 8) &
        df['category'].isin(ALLOWED_CATEGORIES)
    )
    clean_df = df[mask][['Customer Remarks', 'category']].rename(
        columns={'Customer Remarks': 'text'}
    )
    clean_df['text'] = clean_df['text'].astype(str).str.strip()
    return clean_df.drop_duplicates(subset=['text']).reset_index(drop=True)


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
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
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

    metrics_data.update({
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "model": "TF-IDF + Logistic Regression"
    })

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train()
