"""model/train_category_model.py
Supervised customer support category classification pipeline.
Fixes:
- Deduplicates real data before splitting.
- Stratified train/test split on real held-out data (no leakage in test set).
- Hand-crafted domain phrases oversampled ONLY in the training split.
- Evaluates candidate models, selects best by weighted F1, refits that model.
- Saves end-to-end sklearn Pipeline to category_pipeline.pkl.
- Writes comprehensive metrics.json (single source of truth).
"""
import os
import sys
import json
import pickle
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure root dir is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_engine.config import ALLOWED_CATEGORIES
from ai_engine.rules import rule_based_category

DATA_PATH = os.path.join(BASE_DIR, "data", "Customer_support_data.csv")
PHRASES_PATH = os.path.join(BASE_DIR, "model", "data", "domain_phrases.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
TRAINING_CSV_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GENERIC_FEEDBACK = {
    'good', 'very good', 'excellent', 'bad', 'very bad', 'nice', 'thanks',
    'thank you', 'ok', 'okay', 'great', 'osm', 'best', 'nothing', 'no',
    'fine', 'na', 'none', 'satisfactory', 'yes', 'super', 'well done',
    'godd job', 'happy'
}


def load_and_curate_data():
    """Load raw dataset, filter substantive remarks, and deduplicate real remarks."""
    print(f"[LOAD] Reading raw data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    raw_count = len(df)

    mask = (
        df['Customer Remarks'].notnull() &
        ~df['Customer Remarks'].astype(str).str.strip().str.lower().isin(GENERIC_FEEDBACK) &
        (df['Customer Remarks'].astype(str).str.strip().str.len() > 8) &
        df['category'].isin(ALLOWED_CATEGORIES)
    )
    substantive_df = df[mask][['Customer Remarks', 'category']].rename(
        columns={'Customer Remarks': 'text'}
    )
    substantive_df['text'] = substantive_df['text'].astype(str).str.strip()

    # Deduplicate real remarks FIRST to avoid train/test duplication
    real_deduped = substantive_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"[DATA] Raw: {raw_count:,} | Substantive: {len(substantive_df):,} | Deduped Real: {len(real_deduped):,}")
    return raw_count, real_deduped


def augment_training_split(train_df, domain_corpus_path):
    """Oversample domain-specific phrases strictly in the training split."""
    with open(domain_corpus_path, 'r', encoding='utf-8') as f:
        domain_corpus = json.load(f)

    records = train_df.to_dict('records')
    for cat, phrases in domain_corpus.items():
        if cat in ALLOWED_CATEGORIES:
            for p in phrases:
                records.extend([{'text': p, 'category': cat}] * 30)

    augmented_train = pd.DataFrame(records).reset_index(drop=True)
    print(f"[AUGMENT] Training records with domain phrases: {len(augmented_train):,}")
    return augmented_train


def train_and_evaluate():
    raw_rows, real_df = load_and_curate_data()

    # Stratified Train/Test split on real data only (graceful fallback if classes are small)
    counts = real_df['category'].value_counts()
    stratify = real_df['category'] if counts.min() >= 2 else None

    train_real, test_real = train_test_split(
        real_df, test_size=0.20, random_state=42, stratify=stratify
    )
    print(f"[SPLIT] Real Train: {len(train_real):,} | Real Test (Held-out): {len(test_real):,}")

    # Augment ONLY the training split
    train_augmented = augment_training_split(train_real, PHRASES_PATH)
    train_augmented.to_csv(TRAINING_CSV_PATH, index=False, encoding='utf-8')

    X_train = train_augmented['text']
    y_train = train_augmented['category']
    X_test = test_real['text']
    y_test = test_real['category']

    # Candidate models
    candidates = {
        "Logistic Regression": LogisticRegression(C=3.0, max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=25, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2)
    }

    results = {}
    trained_pipelines = {}

    print("\n[EVALUATION] Training and evaluating candidate pipelines on held-out real remarks...")
    for name, clf in candidates.items():
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, stop_words='english')),
            ('clf', clf)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average='weighted', zero_division=0))
        prec = float(precision_score(y_test, preds, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, preds, average='weighted', zero_division=0))

        results[name] = {
            "Accuracy": round(acc, 4),
            "F1 Score": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4)
        }
        trained_pipelines[name] = pipe
        print(f"  {name:25} -> Acc: {acc*100:.2f}% | F1: {f1:.4f}")

    # Select best model by weighted F1
    best_model_name = max(results, key=lambda k: results[k]["F1 Score"])
    best_metrics = results[best_model_name]
    print(f"\n[BEST MODEL] Selected '{best_model_name}' by weighted F1 ({best_metrics['F1 Score']:.4f})")

    # Refit THAT selected model on all training data
    best_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, stop_words='english')),
        ('clf', candidates[best_model_name])
    ])
    best_pipeline.fit(X_train, y_train)

    # Hybrid evaluation on held-out test set
    rule_hits = 0
    ml_hits = 0
    hybrid_preds = []

    for text, true_cat in zip(X_test, y_test):
        rule_cat = rule_based_category(text)
        if rule_cat != 'Others':
            hybrid_preds.append(rule_cat)
            rule_hits += 1
        else:
            ml_pred = best_pipeline.predict([text])[0]
            hybrid_preds.append(ml_pred)
            ml_hits += 1

    hybrid_acc = float(accuracy_score(y_test, hybrid_preds))
    hybrid_f1 = float(f1_score(y_test, hybrid_preds, average='weighted', zero_division=0))
    total_test = len(y_test)
    rule_pct = round((rule_hits / total_test) * 100, 2)
    ml_pct = round((ml_hits / total_test) * 100, 2)

    print(f"\n[HYBRID] End-to-end Test Accuracy: {hybrid_acc*100:.2f}% | F1: {hybrid_f1:.4f}")
    print(f"[HYBRID] Triage decision split: Rules: {rule_hits} ({rule_pct}%) | ML Fallback: {ml_hits} ({ml_pct}%)")

    # Save artifacts
    pipeline_path = os.path.join(MODEL_DIR, "category_pipeline.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(best_pipeline, f)

    # Also save individual components for backwards compatibility
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(best_pipeline.named_steps['tfidf'], f)
    with open(os.path.join(MODEL_DIR, "category_model.pkl"), "wb") as f:
        pickle.dump(best_pipeline.named_steps['clf'], f)

    # Load existing metrics or create fresh
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding='utf-8') as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    metrics_data.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": {
            "raw_dataset_rows": raw_rows,
            "substantive_remarks": len(real_df),
            "train_set_size": len(train_augmented),
            "held_out_test_size": total_test
        },
        "category_model": {
            "selected_model": best_model_name,
            "ml_only": {
                "accuracy": best_metrics["Accuracy"],
                "weighted_f1": best_metrics["F1 Score"],
                "precision": best_metrics["Precision"],
                "recall": best_metrics["Recall"]
            },
            "hybrid_end_to_end": {
                "accuracy": round(hybrid_acc, 4),
                "weighted_f1": round(hybrid_f1, 4)
            },
            "decision_split": {
                "rules_resolved_count": rule_hits,
                "rules_resolved_pct": rule_pct,
                "ml_fallback_count": ml_hits,
                "ml_fallback_pct": ml_pct
            },
            "candidate_models": results
        }
    })

    with open(METRICS_PATH, "w", encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)

    # Legacy output text files
    with open(os.path.join(OUTPUT_DIR, "category_accuracy.txt"), "w") as f:
        f.write(str(best_metrics["Accuracy"]))
    with open(os.path.join(OUTPUT_DIR, "model_comparison_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[OK] Saved artifacts to {MODEL_DIR} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    train_and_evaluate()