"""model/eda.py
Exploratory Data Analysis: text length stats and top n-grams on cleaned customer support data.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "model", "output", "cleaned_customer_support_data.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "data", "new_Customer_support_data.csv")

df = pd.read_csv(CSV_PATH, encoding='latin1' if 'new_' in CSV_PATH else 'utf-8', low_memory=False)

if 'text' in df.columns:
    text_col = 'text'
elif 'Customer Remarks' in df.columns:
    text_col = 'Customer Remarks'
else:
    df['text'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).str.strip()
    text_col = 'text'

df = df[df[text_col].notnull()].copy()
df['text_len'] = df[text_col].astype(str).str.len()

print(f"[EDA] Records: {len(df):,} | Length: Mean={df['text_len'].mean():.1f}, Min={df['text_len'].min()}, Max={df['text_len'].max()}")

cat_col = 'category' if 'category' in df.columns else 'queue'
if cat_col in df.columns:
    print(f"\n[EDA] Category ({cat_col}) breakdown:\n", df[cat_col].value_counts())

if 'priority' in df.columns:
    print("\n[EDA] Priority breakdown:\n", df['priority'].value_counts())


def top_ngrams(corpus, n=1, top_k=8):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    counts = vec.transform(corpus).sum(axis=0)
    words = sorted([(w, counts[0, idx]) for w, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)
    return words[:top_k]


corpus = df[text_col].dropna().astype(str).tolist()
print("\n[EDA] Top 1-grams:", top_ngrams(corpus, n=1))
print("[EDA] Top 2-grams:", top_ngrams(corpus, n=2))
print("[EDA] Completed successfully.")
