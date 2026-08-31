# model/topic_modelling_nmf.py
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

print("="*70)
print("[STEP 4 & 5] FEATURE EXTRACTION & TOPIC MODELLING USING NMF")
print("="*70)

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_tickets.csv")

# 1. Check & load preprocessed data
if not os.path.exists(CLEANED_DATA_PATH) or os.path.getsize(CLEANED_DATA_PATH) == 0:
    print("[INFO] Cleaned data not found. Running preprocessing first...")
    from preprocessing.text_preprocessing import df_clean
else:
    print(f"[LOAD] Loading cleaned tickets from: {CLEANED_DATA_PATH}")
    df_clean = pd.read_csv(CLEANED_DATA_PATH)

print(f"[DATA] Dataset size: {len(df_clean)} records")

# Ensure valid text
if 'Complaint_clean' not in df_clean.columns:
    df_clean['Complaint_clean'] = df_clean['complaint_what_happened']

df_clean['Complaint_clean'] = df_clean['Complaint_clean'].fillna('').astype(str)
df_clean = df_clean[df_clean['Complaint_clean'].str.strip().str.len() > 2].reset_index(drop=True)

# -------------------------------------------------------------------
# STEP 4: FEATURE EXTRACTION (TF-IDF)
# -------------------------------------------------------------------
print("\n[TF-IDF] Initialising TfidfVectorizer (max_df=0.98, min_df=2, stop_words='english')...")
tf_idf_vec = TfidfVectorizer(max_df=0.98, min_df=2, stop_words='english')

# Document Term Matrix
print("[TF-IDF] Creating Document Term Matrix...")
tfidf = tf_idf_vec.fit_transform(df_clean['Complaint_clean'])
feature_names = tf_idf_vec.get_feature_names_out()
print(f"[OK] TF-IDF Matrix shape: {tfidf.shape} (Vocabulary size: {len(feature_names)})")

# -------------------------------------------------------------------
# STEP 5: TOPIC MODELLING USING NMF (5 TOPICS)
# -------------------------------------------------------------------
num_topics = 5
print(f"\n[NMF] Training NMF Model with {num_topics} components (random_state=40)...")
nmf_model = NMF(n_components=num_topics, random_state=40)
nmf_model.fit(tfidf)

print("\n" + "="*70)
print("[TOPICS] TOP 15 WORDS FOR EACH TOPIC")
print("="*70)
for index, topic in enumerate(nmf_model.components_):
    top_indices = topic.argsort()[-15:]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\n* Topic #{index}:")
    print("   " + ", ".join(top_words))

# Assign best topic to each complaint
print("\n[MAPPING] Assigning best topic to each complaint...")
topic_values = nmf_model.transform(tfidf)
df_clean['Topic'] = topic_values.argmax(axis=1)

# Map topics to standard business categories
Topic_names = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}
df_clean['Topic_category'] = df_clean['Topic'].map(Topic_names)

print("\n[DISTRIBUTION] Topic Distribution:")
for topic_id, name in Topic_names.items():
    count = (df_clean['Topic'] == topic_id).sum()
    pct = (count / len(df_clean)) * 100 if len(df_clean) > 0 else 0
    print(f"  [{topic_id}] {name:30} : {count:5d} ({pct:5.1f}%)")

# -------------------------------------------------------------------
# SAVE ARTIFACTS AND LABELED TRAINING DATA
# -------------------------------------------------------------------
training_data = df_clean[['complaint_what_happened', 'Topic', 'Topic_category', 'Complaint_clean']].copy()
training_data_path = os.path.join(OUTPUT_DIR, "training_data.csv")
training_data.to_csv(training_data_path, index=False, encoding='utf-8')
print(f"\n[SAVE] Saved labeled training data to: {training_data_path}")

# Save NMF Model and TF-IDF Vectorizer
with open(os.path.join(MODEL_DIR, "nmf_model.pkl"), "wb") as f:
    pickle.dump(nmf_model, f)

with open(os.path.join(MODEL_DIR, "nmf_tfidf.pkl"), "wb") as f:
    pickle.dump(tf_idf_vec, f)

print(f"[OK] Saved NMF model and TF-IDF vectorizer to {MODEL_DIR}")

print("\n" + "="*70)
print("[SUCCESS] TOPIC MODELLING COMPLETED SUCCESSFULLY!")
print("="*70)
