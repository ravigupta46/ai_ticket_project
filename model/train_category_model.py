# model/train_category_model.py
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Optional XGBoost import with graceful fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGBOOST = False

print("="*80)
print("[STEP 6 & 7] SUPERVISED MODEL BUILDING, TRAINING & EVALUATION")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")

# 1. Load Training Data
if not os.path.exists(TRAINING_DATA_PATH) or os.path.getsize(TRAINING_DATA_PATH) == 0:
    print("[INFO] Training data not found. Running topic modeling first...")
    from model.topic_modelling_nmf import training_data
else:
    print(f"[LOAD] Loading labeled training data from: {TRAINING_DATA_PATH}")
    training_data = pd.read_csv(TRAINING_DATA_PATH)

# Validate columns
text_col = 'complaint_what_happened' if 'complaint_what_happened' in training_data.columns else 'Complaint_clean'
topic_col = 'Topic'

training_data = training_data.dropna(subset=[text_col, topic_col]).reset_index(drop=True)
print(f"[DATA] Total training samples: {len(training_data)}")

Topic_names = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}

# -------------------------------------------------------------------
# FEATURE EXTRACTION FOR SUPERVISED LEARNING
# -------------------------------------------------------------------
print("\n[INFO] Extracting Count Vectors & TF-IDF Transformation...")
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data[text_col])

tfidf_transformer = TfidfTransformer()
X_train_tf = tfidf_transformer.fit_transform(X_train_counts)

# Also prepare a standalone TF-IDF Vectorizer for simple unified deployment
tfidf_unified = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_unified = tfidf_unified.fit_transform(training_data[text_col])

# Train-test split (80/20, random_state=40, stratified)
train_X, test_X, train_y, test_y = train_test_split(
    X_train_tf, training_data[topic_col], test_size=0.2, random_state=40, stratify=training_data[topic_col]
)

print(f"[DATA] Training shape: {train_X.shape}, Test shape: {test_X.shape}")

# -------------------------------------------------------------------
# MODEL EVALUATION HELPER
# -------------------------------------------------------------------
def eval_model(y_true, y_pred, y_pred_proba, model_name="Model", stage="Test"):
    """Compute and print all classification metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
        else:
            auc = 0.0
    except Exception:
        auc = 0.0
        
    print(f"\n--- {model_name} [{stage} Results] ---")
    print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (W):   {prec:.4f}")
    print(f"Recall (W):      {rec:.4f}")
    print(f"F1-Score (W):    {f1:.4f}")
    print(f"ROC AUC (W-OvR): {auc:.4f}")
    
    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1 Score": round(float(f1), 4),
        "ROC AUC Score": round(float(auc), 4)
    }

# -------------------------------------------------------------------
# CANDIDATE MODELS TRAINING & COMPARISON
# -------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=40, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=40, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=40, class_weight='balanced', n_jobs=-1),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    "XGBoost Classifier": XGBClassifier(random_state=40)
}

results = {}
trained_models = {}

print("\n" + "="*80)
print("[TRAINING] TRAINING AND EVALUATING CANDIDATE MODELS")
print("="*80)

for name, clf in models.items():
    print(f"\n* Training {name}...")
    try:
        clf.fit(train_X, train_y)
        trained_models[name] = clf
        
        # Predictions
        y_test_pred = clf.predict(test_X)
        y_test_proba = clf.predict_proba(test_X) if hasattr(clf, "predict_proba") else None
        
        metrics = eval_model(test_y, y_test_pred, y_test_proba, model_name=name, stage="Test")
        results[name] = metrics
    except Exception as e:
        print(f"[ERROR] Error training {name}: {e}")

# -------------------------------------------------------------------
# FINAL MODEL COMPARISON TABLE
# -------------------------------------------------------------------
print("\n" + "="*80)
print("[SUMMARY] FINAL MODEL COMPARISON SUMMARY")
print("="*80)
results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Select best model by F1 Score
best_model_name = results_df['F1 Score'].idxmax()
best_model = trained_models[best_model_name]
best_accuracy = results[best_model_name]["Accuracy"]

print("\n" + "="*80)
print(f"[BEST MODEL] {best_model_name}")
print(f"[PERFORMANCE] Test Accuracy: {best_accuracy*100:.2f}% | Weighted F1: {results[best_model_name]['F1 Score']}")
print("="*80 + "\n")

# Detailed Classification Report for Best Model
best_pred = best_model.predict(test_X)
print("\n[REPORT] Detailed Classification Report for Best Model:")
target_names = [Topic_names[i] for i in sorted(Topic_names.keys())]
print(classification_report(test_y, best_pred, target_names=target_names, zero_division=0))

# -------------------------------------------------------------------
# SERIALIZE BEST MODEL AND VECTORIZERS
# -------------------------------------------------------------------
print("\n[SAVE] Saving best model and artifacts...")

# Fit unified pipeline model for production inference
best_unified_model = LogisticRegression(C=1.0, max_iter=1000, random_state=40, class_weight='balanced')
best_unified_model.fit(X_unified, training_data[topic_col])

# Save vectorizers
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf_unified, f)

with open(os.path.join(MODEL_DIR, "count_vectorizer.pkl"), "wb") as f:
    pickle.dump(count_vect, f)

with open(os.path.join(MODEL_DIR, "tfidf_transformer.pkl"), "wb") as f:
    pickle.dump(tfidf_transformer, f)

# Save category model
with open(os.path.join(MODEL_DIR, "category_model.pkl"), "wb") as f:
    pickle.dump(best_unified_model, f)

# Save accuracy txt
with open(os.path.join(OUTPUT_DIR, "category_accuracy.txt"), "w") as f:
    f.write(str(best_accuracy))

# Save results JSON
with open(os.path.join(OUTPUT_DIR, "model_comparison_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"[OK] Saved category_model.pkl, vectorizers, and metrics to {MODEL_DIR}")

# -------------------------------------------------------------------
# STEP 8: INFERENCE VERIFICATION
# -------------------------------------------------------------------
print("\n" + "="*80)
print("[STEP 8] INFERENCE ON TEST SAMPLES")
print("="*80)

sample_complaints = [
    "I can not get from chase who services my mortgage, who owns it and who has original loan docs",
    "The bill amount of my credit card was debited twice. Please look into the matter and resolve at the earliest.",
    "I want to open a salary account at your downtown branch. Please provide me the procedure.",
    "Yesterday, I received a fraudulent email regarding renewal of my services.",
    "What is the procedure to know my CIBIL score?",
    "I need to know the number of bank branches and their locations in the city of Dubai"
]

def predict_category(text_list):
    transformed = tfidf_unified.transform(text_list)
    preds = best_unified_model.predict(transformed)
    return [Topic_names[p] for p in preds]

predictions = predict_category(sample_complaints)
for text, pred in zip(sample_complaints, predictions):
    print(f"\n[Complaint]: {text}")
    print(f" -> [Predicted Category]: {pred}")

print("\n" + "="*80)
print("[SUCCESS] MODEL TRAINING & INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)