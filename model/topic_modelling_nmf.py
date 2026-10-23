"""model/topic_modelling_nmf.py
Unsupervised NMF topic extraction for customer support complaints.
"""
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CSV_PATH = os.path.join(MODEL_DIR, "output", "training_data.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "data", "Customer_support_data.csv")

df = pd.read_csv(CSV_PATH, low_memory=False)
text_col = 'text' if 'text' in df.columns else 'Customer Remarks'
corpus = df[text_col].dropna().astype(str)
corpus = corpus[corpus.str.len() > 8].reset_index(drop=True)

tfidf_vec = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=10000)
tfidf = tfidf_vec.fit_transform(corpus)

num_topics = 5
nmf = NMF(n_components=num_topics, random_state=42)
nmf.fit(tfidf)

feature_names = tfidf_vec.get_feature_names_out()
for idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    print(f"Topic #{idx}: {', '.join(top_words)}")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "nmf_model.pkl"), "wb") as f:
    pickle.dump(nmf, f)
with open(os.path.join(MODEL_DIR, "nmf_tfidf.pkl"), "wb") as f:
    pickle.dump(tfidf_vec, f)

print("[NMF] Saved nmf_model.pkl and nmf_tfidf.pkl successfully.")
