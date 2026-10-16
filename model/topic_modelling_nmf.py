# model/topic_modelling_nmf.py
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

print("=" * 70)
print("[STEP 4 & 5] FEATURE EXTRACTION & TOPIC MODELLING ON CUSTOMER DATA")
print("=" * 70)

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "Customer_support_data.csv")

if os.path.exists(TRAINING_DATA_PATH):
    print(f"[LOAD] Loading training data from: {TRAINING_DATA_PATH}")
    df_clean = pd.read_csv(TRAINING_DATA_PATH)
    text_col = 'text' if 'text' in df_clean.columns else 'Complaint_clean'
else:
    print(f"[LOAD] Loading customer data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    df_clean = df[df['Customer Remarks'].notnull()].copy()
    text_col = 'Customer Remarks'

df_clean[text_col] = df_clean[text_col].fillna('').astype(str)
df_clean = df_clean[df_clean[text_col].str.strip().str.len() > 3].reset_index(drop=True)
print(f"[DATA] Usable customer records: {len(df_clean):,}")

# -------------------------------------------------------------------
# FEATURE EXTRACTION (TF-IDF)
# -------------------------------------------------------------------
print("\n[TF-IDF] Fitting TF-IDF Vectorizer...")
tf_idf_vec = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=10000)
tfidf = tf_idf_vec.fit_transform(df_clean[text_col])
feature_names = tf_idf_vec.get_feature_names_out()
print(f"[OK] TF-IDF Matrix shape: {tfidf.shape}")

# -------------------------------------------------------------------
# TOPIC MODELLING USING NMF
# -------------------------------------------------------------------
num_topics = 5
print(f"\n[NMF] Training NMF Model with {num_topics} customer support topics...")
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(tfidf)

print("\n" + "=" * 70)
print("[TOPICS] TOP 15 WORDS FOR EACH TOPIC")
print("=" * 70)
for index, topic in enumerate(nmf_model.components_):
    top_indices = topic.argsort()[-15:]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\n* Topic #{index}:")
    print("   " + ", ".join(top_words))

# Map discovered topics to customer service domains
Topic_names = {
    0: 'Returns & Exchange',
    1: 'Order & Delivery Tracking',
    2: 'Product Queries & Technical',
    3: 'Payments & Refunds',
    4: 'Shopzilla Account & Support'
}

topic_values = nmf_model.transform(tfidf)
df_clean['Topic'] = topic_values.argmax(axis=1)
df_clean['Topic_category'] = df_clean['Topic'].map(Topic_names)

print("\n[DISTRIBUTION] Topic Distribution:")
for topic_id, name in Topic_names.items():
    count = (df_clean['Topic'] == topic_id).sum()
    pct = (count / len(df_clean)) * 100 if len(df_clean) > 0 else 0
    print(f"  [{topic_id}] {name:32} : {count:5d} ({pct:5.1f}%)")

# Save NMF Model and TF-IDF Vectorizer
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "nmf_model.pkl"), "wb") as f:
    pickle.dump(nmf_model, f)

with open(os.path.join(MODEL_DIR, "nmf_tfidf.pkl"), "wb") as f:
    pickle.dump(tf_idf_vec, f)

print(f"[OK] Saved nmf_model.pkl and nmf_tfidf.pkl to {MODEL_DIR}")
print("\n" + "=" * 70)
print("[SUCCESS] TOPIC MODELLING COMPLETED SUCCESSFULLY!")
print("=" * 70)
