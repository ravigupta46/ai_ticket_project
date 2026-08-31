# model/eda.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

print("="*70)
print("[STEP 3] EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_tickets.csv")

if not os.path.exists(CLEANED_DATA_PATH) or os.path.getsize(CLEANED_DATA_PATH) == 0:
    print("[INFO] Cleaned tickets not found. Running preprocessing...")
    from preprocessing.text_preprocessing import df_clean
else:
    df_clean = pd.read_csv(CLEANED_DATA_PATH)

print(f"[DATA] Dataset size: {len(df_clean)} records")

# 1. Complaint Character Length Statistics
print("\n" + "-"*70)
print("[LENGTH] 1. COMPLAINT CHARACTER LENGTH ANALYSIS")
print("-" * 70)

if 'complaint_what_happened' in df_clean.columns:
    df_clean['complaint_length'] = df_clean['complaint_what_happened'].astype(str).str.len()
    print(f"Raw Complaint Length      -> Mean: {df_clean['complaint_length'].mean():.1f} | Min: {df_clean['complaint_length'].min()} | Max: {df_clean['complaint_length'].max()}")

if 'complaint_what_happened_lemmatized' in df_clean.columns:
    df_clean['lemmatized_length'] = df_clean['complaint_what_happened_lemmatized'].astype(str).str.len()
    print(f"Lemmatized Text Length    -> Mean: {df_clean['lemmatized_length'].mean():.1f} | Min: {df_clean['lemmatized_length'].min()} | Max: {df_clean['lemmatized_length'].max()}")

if 'Complaint_clean' in df_clean.columns:
    df_clean['clean_length'] = df_clean['Complaint_clean'].astype(str).str.len()
    print(f"Clean POS Removed Length  -> Mean: {df_clean['clean_length'].mean():.1f} | Min: {df_clean['clean_length'].min()} | Max: {df_clean['clean_length'].max()}")

# 2. Top N-Grams Function
def get_top_n_words(corpus, n=1, count=10):
    """Extract top N-grams by frequency"""
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:count]

corpus = df_clean['Complaint_clean'].fillna('').astype(str).tolist()

print("\n" + "-"*70)
print("[N-GRAMS] 2. TOP UNIGRAMS (1-gram)")
print("-" * 70)
unigrams = get_top_n_words(corpus, n=1, count=10)
for word, freq in unigrams:
    bar = "=" * max(1, int(freq / (unigrams[0][1] if unigrams else 1) * 30))
    print(f"  {word:18} : {freq:6d} {bar}")

print("\n" + "-"*70)
print("[N-GRAMS] 3. TOP BIGRAMS (2-gram)")
print("-" * 70)
bigrams = get_top_n_words(corpus, n=2, count=10)
for word, freq in bigrams:
    bar = "=" * max(1, int(freq / (bigrams[0][1] if bigrams else 1) * 30))
    print(f"  {word:25} : {freq:6d} {bar}")

print("\n" + "-"*70)
print("[N-GRAMS] 4. TOP TRIGRAMS (3-gram)")
print("-" * 70)
trigrams = get_top_n_words(corpus, n=3, count=10)
for word, freq in trigrams:
    bar = "=" * max(1, int(freq / (trigrams[0][1] if trigrams else 1) * 30))
    print(f"  {word:35} : {freq:6d} {bar}")

# 5. WordCloud Generation
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud = WordCloud(background_color='white', width=1200, height=800, max_words=40).generate(' '.join(corpus))
    wordcloud_path = os.path.join(OUTPUT_DIR, "wordcloud.png")
    wordcloud.to_file(wordcloud_path)
    print(f"\n[SAVE] Saved wordcloud to: {wordcloud_path}")
except ImportError:
    print("\n[INFO] WordCloud / Matplotlib optional package not installed. Text frequency analysis completed above.")

print("\n" + "="*70)
print("[SUCCESS] EDA ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*70)

