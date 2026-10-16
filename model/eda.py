# model/eda.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

print("=" * 70)
print("[STEP 3] EXPLORATORY DATA ANALYSIS (EDA) ON CUSTOMER DATA")
print("=" * 70)

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "Customer_support_data.csv")

if os.path.exists(TRAINING_DATA_PATH):
    df_clean = pd.read_csv(TRAINING_DATA_PATH)
    text_col = 'text' if 'text' in df_clean.columns else df_clean.columns[0]
else:
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    df_clean = df[df['Customer Remarks'].notnull()].copy()
    text_col = 'Customer Remarks'

print(f"[DATA] Dataset size: {len(df_clean):,} records")

# 1. Character Length Statistics
print("\n" + "-" * 70)
print("[LENGTH] 1. COMPLAINT CHARACTER LENGTH ANALYSIS")
print("-" * 70)

df_clean['text_length'] = df_clean[text_col].astype(str).str.len()
print(f"Text Length -> Mean: {df_clean['text_length'].mean():.1f} | Min: {df_clean['text_length'].min()} | Max: {df_clean['text_length'].max()}")

if 'category' in df_clean.columns:
    print("\n[CATEGORY DISTRIBUTION]")
    print(df_clean['category'].value_counts())

# 2. Top N-Grams Function
def get_top_n_words(corpus, n=1, count=10):
    """Extract top N-grams by frequency"""
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:count]

corpus = df_clean[text_col].fillna('').astype(str).tolist()

print("\n" + "-" * 70)
print("[N-GRAMS] 2. TOP UNIGRAMS (1-gram)")
print("-" * 70)
unigrams = get_top_n_words(corpus, n=1, count=10)
for word, freq in unigrams:
    bar = "=" * max(1, int(freq / (unigrams[0][1] if unigrams else 1) * 30))
    print(f"  {word:18} : {freq:6d} {bar}")

print("\n" + "-" * 70)
print("[N-GRAMS] 3. TOP BIGRAMS (2-gram)")
print("-" * 70)
bigrams = get_top_n_words(corpus, n=2, count=10)
for word, freq in bigrams:
    bar = "=" * max(1, int(freq / (bigrams[0][1] if bigrams else 1) * 30))
    print(f"  {word:25} : {freq:6d} {bar}")

print("\n" + "=" * 70)
print("[SUCCESS] EDA ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 70)
