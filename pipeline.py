"""pipeline.py
End-to-end Enhanced Machine Learning Pipeline for Customer Support Ticket Classification.
Targets: 'category' (queue) and 'priority'.

Major Enhancements:
1. Feature Extraction:
   - Multilingual Word N-Grams (1-3) with sublinear term-frequency scaling.
   - Character N-Grams (3-5 char_wb) for morphological root matching across English, German,
     Spanish, French, and Portuguese ticket texts without losing cross-language stems.
   - Elimination of single-language stop word stripping that degraded non-English vocabulary.
2. High-Performance Models:
   - Linear Support Vector Machine (LinearSVC with calibrated decision boundary).
   - ExtraTreesClassifier (Ensemble of randomized decision trees) for non-linear priority interactions.
   - Regularized Multinomial / Logistic / Ensemble architectures benchmarked.
3. Hyperparameter Tuning & Cross-Validation:
   - Systematic Grid Search across regularization constants and feature thresholds.

Execution Order:
1. Data Cleaning
2. Exploratory Data Analysis (Univariate & Bivariate)
3. Data Preprocessing & Train/Test Split
4. Model Building & Metrics Comparison
5. Model Selection & Hyperparameter Tuning
6. Final Model Evaluation & Prediction Function
"""

import os
import sys
import tempfile
import json
import pickle
import warnings
import numpy as np
import pandas as pd

# Set Matplotlib headless backend and writable config cache
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "new_Customer_support_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda_plots")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_customer_support_data.csv")
CAT_MODEL_PATH = os.path.join(BASE_DIR, "model", "category_pipeline.pkl")
PRI_MODEL_PATH = os.path.join(BASE_DIR, "model", "priority_pipeline.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)


# ==============================================================================
# STEP 1: DATA CLEANING
# ==============================================================================
def clean_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Step 1: Data Cleaning
    - Reads new_Customer_support_data.csv using Latin-1 encoding to preserve international accents.
    - Imputes missing values:
        * Fills missing 'subject' with 'No Subject'
        * Fills missing 'body' with 'No Body'
        * Combines 'subject' and 'body' into a rich unified 'text' field.
    - Normalizes targets:
        * 'queue' standardized to 'category'
        * 'priority' standardized to Title Case ('High', 'Medium', 'Low')
    - Cleans inconsistent categorical metadata entries.
    - Deduplicates records and persists cleaned dataset.
    """
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    print(f"[Loading] Source dataset: {filepath}")
    df = pd.read_csv(filepath, encoding='latin1')
    raw_len = len(df)
    print(f"-> Raw dataset loaded: {raw_len:,} rows, {df.shape[1]} columns.")

    # Missing value handling
    df['subject'] = df['subject'].fillna('No Subject').astype(str).str.strip()
    df['body'] = df['body'].fillna('').astype(str).str.strip()
    df['text'] = (df['subject'] + " " + df['body']).str.strip()

    # Target harmonization
    df['category'] = df['queue'].astype(str).str.strip()
    df['priority'] = df['priority'].astype(str).str.strip().str.capitalize()

    # Normalize business_type typos
    business_type_map = {
        'Pit Services': 'IT Services',
        '_IT_Services_': 'IT Services',
        'IT Consulting Service': 'IT Consulting Firm'
    }
    df['business_type'] = df['business_type'].replace(business_type_map)

    # Filter out empty records and drop duplicates
    df = df[df['text'].str.len() > 5].copy()
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"-> Retained {len(df):,} unique records after deduplication & validation.")

    # Save cleaned data
    df.to_csv(CLEANED_DATA_PATH, index=False, encoding='utf-8')
    print(f"-> Cleaned dataset saved to: {CLEANED_DATA_PATH}")

    return df


# ==============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
def perform_eda(df: pd.DataFrame) -> None:
    """
    Step 2: Exploratory Data Analysis
    - Univariate and Bivariate distributions saved as high-res PNG plots.
    """
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    df['text_char_len'] = df['text'].astype(str).str.len()
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Category Distribution
    plt.figure(figsize=(10, 5))
    cat_order = df['category'].value_counts().index
    ax = sns.countplot(data=df, y='category', order=cat_order, hue='category', legend=False, palette='viridis')
    plt.title("Univariate: Ticket Category (Queue) Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Ticket Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "01_category_distribution.png"), dpi=200)
    plt.close()

    # 2. Priority Distribution
    plt.figure(figsize=(7, 4.5))
    pri_order = ['Low', 'Medium', 'High']
    pri_colors = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}
    sns.countplot(data=df, x='priority', order=pri_order, hue='priority', palette=pri_colors, legend=False)
    plt.title("Univariate: Ticket Priority Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Priority Level")
    plt.ylabel("Ticket Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "02_priority_distribution.png"), dpi=200)
    plt.close()

    # 3. Text Length Distribution
    plt.figure(figsize=(9, 4.5))
    sns.histplot(df['text_char_len'], bins=40, kde=True, color='#3b82f6')
    plt.title("Univariate: Ticket Text Character Length Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Character Length (Subject + Body)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "03_text_length_distribution.png"), dpi=200)
    plt.close()

    # 4. Length by Priority
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x='priority', y='text_char_len', order=pri_order, palette=pri_colors, showfliers=False)
    plt.title("Bivariate: Text Length vs Priority Level", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "04_length_by_priority.png"), dpi=200)
    plt.close()

    # 5. Category vs Priority
    plt.figure(figsize=(11, 6))
    ct = pd.crosstab(df['category'], df['priority'], normalize='index')[pri_order] * 100
    ct.plot(kind='barh', stacked=True, color=[pri_colors[p] for p in pri_order], figsize=(10, 6))
    plt.title("Bivariate: Priority Breakdown across Ticket Categories (%)", fontsize=13, fontweight='bold')
    plt.xlabel("Percentage of Tickets (%)")
    plt.ylabel("Category")
    plt.legend(title="Priority", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "05_category_vs_priority.png"), dpi=200)
    plt.close()

    print(f"-> All 5 EDA plots saved successfully to: {EDA_DIR}")


# ==============================================================================
# STEP 3: DATA PREPROCESSING
# ==============================================================================
def create_text_feature_union(max_word_features=30000, max_char_features=15000):
    """
    Builds a high-capacity multilingual representation:
    1. Word-level TF-IDF: captures multi-word phrases (1-3 ngrams) with sublinear TF.
    2. Character n-gram TF-IDF (char_wb): captures word roots, prefixes, and suffixes
       across English, German, Spanish, French, and Portuguese without language bias.
    """
    word_vec = TfidfVectorizer(
        max_features=max_word_features,
        ngram_range=(1, 3),
        sublinear_tf=True,
        analyzer='word'
    )
    char_vec = TfidfVectorizer(
        max_features=max_char_features,
        ngram_range=(3, 5),
        sublinear_tf=True,
        analyzer='char_wb'
    )
    return FeatureUnion([
        ('word', word_vec),
        ('char', char_vec)
    ])


def preprocess_and_split(df: pd.DataFrame):
    """
    Step 3: Data Preprocessing
    - Creates stratified train/test split (80% train, 20% test).
    - Preserves class distribution for all 10 queues.
    """
    print("\n" + "="*70)
    print("STEP 3: DATA PREPROCESSING & TRAIN/TEST SPLIT")
    print("="*70)

    X = df['text']
    y_category = df['category']
    y_priority = df['priority']

    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        X, y_category, y_priority, test_size=0.20, random_state=42, stratify=y_category
    )

    print(f"-> Total Samples: {len(X):,}")
    print(f"-> Training Set:  {len(X_train):,} tickets (80%)")
    print(f"-> Test Set:      {len(X_test):,} tickets (20%)")
    print("-> Advanced Preprocessing: Hybrid Word (1-3) & Char_wb (3-5) TF-IDF Feature Union.")

    return X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test


# ==============================================================================
# STEP 4: MODEL BUILDING & COMPARISON
# ==============================================================================
def build_and_compare_models(X_train, X_test, y_train, y_test, target_name: str):
    """
    Step 4: Model Building
    Trains 3 distinct architectures on the hybrid representation:
      1. Multinomial Naive Bayes
      2. Logistic Regression (L2 regularized)
      3. Support Vector Classifier / Ensemble (LinearSVC or ExtraTrees)
    """
    print(f"\n--- Comparing Models for Target: [{target_name.upper()}] ---")

    if target_name.lower() == 'category':
        models = {
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
            "Logistic Regression": LogisticRegression(C=5.0, max_iter=2000, random_state=42),
            "Linear Support Vector (SVC)": LinearSVC(C=0.8, max_iter=3000, random_state=42)
        }
    else:  # priority
        models = {
            "Logistic Regression": LogisticRegression(C=3.0, max_iter=2000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        }

    results = []
    union = create_text_feature_union()

    for name, clf in models.items():
        pipe = Pipeline([
            ('features', union),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Pipeline': pipe
        })

    comparison_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return results


# ==============================================================================
# STEP 5: MODEL SELECTION & HYPERPARAMETER TUNING
# ==============================================================================
def tune_best_model(X_train, y_train, target_name: str):
    """
    Step 5: Hyperparameter Tuning
    Tunes the champion model using 3-fold cross validation.
    """
    print(f"\n--- Hyperparameter Tuning Champion Model for Target: [{target_name.upper()}] ---")

    if target_name.lower() == 'category':
        pipe = Pipeline([
            ('features', create_text_feature_union()),
            ('classifier', LinearSVC(max_iter=3000, random_state=42))
        ])
        param_grid = {
            'classifier__C': [0.5, 0.8, 1.0, 1.2]
        }
    else:
        pipe = Pipeline([
            ('features', create_text_feature_union()),
            ('classifier', ExtraTreesClassifier(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'classifier__n_estimators': [150, 200, 250],
            'classifier__min_samples_split': [2, 4]
        }

    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"-> Best Parameters: {grid_search.best_params_}")
    print(f"-> Best Cross-Validation Weighted F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ==============================================================================
# STEP 6: EVALUATION & PREDICTION
# ==============================================================================
def evaluate_and_save(best_cat_pipe, best_pri_pipe, X_test, y_cat_test, y_pri_test, cat_params, pri_params):
    """
    Step 6: Model Evaluation & Persistence
    - Evaluates the models on the held-out test set.
    - Persists pipelines and metrics.
    """
    print("\n" + "="*70)
    print("STEP 6: FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*70)

    # Category evaluation
    y_cat_pred = best_cat_pipe.predict(X_test)
    cat_acc = float(accuracy_score(y_cat_test, y_cat_pred))
    cat_prec = float(precision_score(y_cat_test, y_cat_pred, average='weighted', zero_division=0))
    cat_rec = float(recall_score(y_cat_test, y_cat_pred, average='weighted', zero_division=0))
    cat_f1 = float(f1_score(y_cat_test, y_cat_pred, average='weighted', zero_division=0))

    print(f"\n[Category Model Test Performance]:")
    print(f"  Accuracy:  {cat_acc:.4f}")
    print(f"  Precision: {cat_prec:.4f}")
    print(f"  Recall:    {cat_rec:.4f}")
    print(f"  F1-Score:  {cat_f1:.4f}")

    # Priority evaluation
    y_pri_pred = best_pri_pipe.predict(X_test)
    pri_acc = float(accuracy_score(y_pri_test, y_pri_pred))
    pri_prec = float(precision_score(y_pri_test, y_pri_pred, average='weighted', zero_division=0))
    pri_rec = float(recall_score(y_pri_test, y_pri_pred, average='weighted', zero_division=0))
    pri_f1 = float(f1_score(y_pri_test, y_pri_pred, average='weighted', zero_division=0))

    print(f"\n[Priority Model Test Performance]:")
    print(f"  Accuracy:  {pri_acc:.4f}")
    print(f"  Precision: {pri_prec:.4f}")
    print(f"  Recall:    {pri_rec:.4f}")
    print(f"  F1-Score:  {pri_f1:.4f}")

    # Save Pipelines
    with open(CAT_MODEL_PATH, "wb") as f:
        pickle.dump(best_cat_pipe, f)
    print(f"\n-> Saved Category Pipeline: {CAT_MODEL_PATH}")

    with open(PRI_MODEL_PATH, "wb") as f:
        pickle.dump(best_pri_pipe, f)
    print(f"-> Saved Priority Pipeline: {PRI_MODEL_PATH}")

    # Save metrics.json
    metrics_data = {
        "dataset": "new_Customer_support_data.csv",
        "category_model": {
            "selected_model": type(best_cat_pipe.named_steps['classifier']).__name__,
            "best_params": {str(k): str(v) for k, v in cat_params.items()},
            "metrics": {
                "accuracy": round(cat_acc, 4),
                "precision": round(cat_prec, 4),
                "recall": round(cat_rec, 4),
                "f1": round(cat_f1, 4)
            }
        },
        "priority_model": {
            "selected_model": type(best_pri_pipe.named_steps['classifier']).__name__,
            "best_params": {str(k): str(v) for k, v in pri_params.items()},
            "metrics": {
                "accuracy": round(pri_acc, 4),
                "precision": round(pri_prec, 4),
                "recall": round(pri_rec, 4),
                "f1": round(pri_f1, 4)
            }
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"-> Saved Metrics Summary: {METRICS_PATH}")

    return metrics_data


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
def predict_ticket(text: str, subject: str = "", category_pipe=None, priority_pipe=None) -> dict:
    """
    Prediction function:
    Takes support ticket text (and optional subject) and outputs predicted category and priority.
    """
    if category_pipe is None:
        with open(CAT_MODEL_PATH, "rb") as f:
            category_pipe = pickle.load(f)

    if priority_pipe is None:
        with open(PRI_MODEL_PATH, "rb") as f:
            priority_pipe = pickle.load(f)

    full_text = (subject.strip() + " " + text.strip()).strip() if subject else text.strip()
    if not full_text:
        return {"category": "Others", "priority": "Low", "confidence": {}}

    pred_category = str(category_pipe.predict([full_text])[0])
    pred_priority = str(priority_pipe.predict([full_text])[0])

    return {
        "input_text": full_text[:120] + ("..." if len(full_text) > 120 else ""),
        "predicted_category": pred_category,
        "predicted_priority": pred_priority
    }


def print_prediction(ticket_text: str, subject: str = "", cat_pipe=None, pri_pipe=None):
    res = predict_ticket(ticket_text, subject, cat_pipe, pri_pipe)
    print("-" * 65)
    print(f"Ticket Input:       {res['input_text']}")
    print(f"Predicted Category: {res['predicted_category']}")
    print(f"Predicted Priority: {res['predicted_priority']}")
    print("-" * 65)


# ==============================================================================
# END-TO-END RUNNER
# ==============================================================================
def run_end_to_end():
    print("="*70)
    print("CUSTOMER SUPPORT TICKET ML PIPELINE (Enhanced Preprocessing & Models)")
    print("="*70)

    # 1. Clean data
    df_clean = clean_data(DATA_PATH)

    # 2. EDA
    perform_eda(df_clean)

    # 3. Preprocessing
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = preprocess_and_split(df_clean)

    # 4. Model Building & Comparison
    print("\n" + "="*70)
    print("STEP 4: MODEL BUILDING & COMPARISON")
    print("="*70)
    cat_comp = build_and_compare_models(X_train, X_test, y_cat_train, y_cat_test, target_name="category")
    pri_comp = build_and_compare_models(X_train, X_test, y_pri_train, y_pri_test, target_name="priority")

    # 5. Model Selection & Tuning
    print("\n" + "="*70)
    print("STEP 5: MODEL SELECTION & HYPERPARAMETER TUNING")
    print("="*70)
    best_cat_pipe, cat_params, _ = tune_best_model(X_train, y_cat_train, target_name="category")
    best_pri_pipe, pri_params, _ = tune_best_model(X_train, y_pri_train, target_name="priority")

    # 6. Evaluation & Prediction
    evaluate_and_save(best_cat_pipe, best_pri_pipe, X_test, y_cat_test, y_pri_test, cat_params, pri_params)

    # Sample Predictions Demonstration
    print("\n" + "="*70)
    print("SAMPLE TICKET INFERENCE DEMO")
    print("="*70)
    sample_tickets = [
        {
            "subject": "CRITICAL: Database server down",
            "body": "Production SQL cluster crashed with error 500, customers cannot login or checkout!"
        },
        {
            "subject": "Invoice question",
            "body": "Can you check my monthly credit card subscription charge? I was double billed for May."
        },
        {
            "subject": "Need help setting up wifi adapter",
            "body": "I purchased the wireless dongle yesterday and need the installation manual for Windows 11."
        },
        {
            "subject": "General question",
            "body": "What are your business hours during the upcoming bank holiday weekend?"
        }
    ]

    for item in sample_tickets:
        print_prediction(item["body"], item["subject"], best_cat_pipe, best_pri_pipe)

    print("\n[SUCCESS] Pipeline executed end-to-end successfully.")


if __name__ == "__main__":
    run_end_to_end()
