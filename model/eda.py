"""model/eda.py
Exploratory Data Analysis: text length stats and top n-grams on curated training data.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "model", "output", "training_data.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "data", "Customer_support_data.csv")

df = pd.read_csv(CSV_PATH, low_memory=False)
text_col = 'text' if 'text' in df.columns else 'Customer Remarks'
df = df[df[text_col].notnull()].copy()
df['text_len'] = df[text_col].astype(str).str.len()

print(f"[EDA] Records: {len(df):,} | Length: Mean={df['text_len'].mean():.1f}, Min={df['text_len'].min()}, Max={df['text_len'].max()}")
if 'category' in df.columns:
    print("\n[EDA] Category breakdown:\n", df['category'].value_counts())

def top_ngrams(corpus, n=1, top_k=8):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    counts = vec.transform(corpus).sum(axis=0)
    words = sorted([(w, counts[0, idx]) for w, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)
    return words[:top_k]

corpus = df[text_col].dropna().astype(str).tolist()
print("\n[EDA] Top 1-grams:", top_ngrams(corpus, n=1))
print("[EDA] Top 2-grams:", top_ngrams(corpus, n=2))
print("[EDA] Completed successfully.")
