# model/priority_model.py
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 70)
print("[PRIORITY] PRIORITY MODEL TRAINING ON CUSTOMER DATA")
print("=" * 70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'model', 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_PATH = os.path.join(OUTPUT_DIR, 'training_data.csv')

# Check training data
if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
    print("[LOAD] Reading from Customer_support_data.csv directly...")
    raw_csv = os.path.join(BASE_DIR, 'data', 'Customer_support_data.csv')
    df = pd.read_csv(raw_csv, low_memory=False)
    text_col = 'Customer Remarks'
else:
    print(f"[LOAD] Loading training data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if 'text' in df.columns:
        text_col = 'text'
    elif 'complaint_what_happened' in df.columns:
        text_col = 'complaint_what_happened'
    else:
        text_col = df.columns[0]

# Determine priority based on customer support urgency signals
def determine_priority(text):
    text = str(text).lower()
    high_keywords = [
        'urgent', 'asap', 'emergency', 'critical', 'immediately', 
        'fraud', 'stolen', 'unauthorized', 'theft', 'skimming', 'lawsuit', 'penalty',
        'fire', 'smoke', 'burst', 'electric shock', 'blast', 'heating very badly',
        'overheating', 'double debited', 'money stolen', 'consumer court', 'police',
        'abusive', 'harassment'
    ]
    medium_keywords = [
        'problem', 'issue', 'difficulty', 'trouble', 'help', 
        'assist', 'delay', 'delayed', 'slow', 'dispute', 'error', 'wrong',
        'defective', 'damaged', 'broken', 'heating', 'not working', 'flickering',
        'refund pending', 'payment failed', 'not delivered', 'missing'
    ]
    
    if any(word in text for word in high_keywords):
        return 'High'
    elif any(word in text for word in medium_keywords):
        return 'Medium'
    else:
        return 'Low'

df['priority'] = df[text_col].apply(determine_priority)
df = df.dropna(subset=[text_col, 'priority']).reset_index(drop=True)

print(f"\n[DISTRIBUTION] Priority Distribution:")
pri_counts = df['priority'].value_counts()
for pri, count in pri_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {pri:8} : {count:5d} ({pct:5.1f}%)")

# Train / Test split
X = df[text_col].astype(str)
y = df['priority']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[DATA] Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# TF-IDF Vectorizer
print("\n[TF-IDF] Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Classifier
print("[TRAINING] Training Random Forest Priority Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print(f"[ACCURACY] PRIORITY MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("=" * 70)

# Save artifacts
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, 'priority_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, 'priority_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(OUTPUT_DIR, 'priority_accuracy.txt'), 'w') as f:
    f.write(str(round(accuracy, 4)))

print(f"[OK] Saved priority_model.pkl, priority_vectorizer.pkl, and priority_accuracy.txt")