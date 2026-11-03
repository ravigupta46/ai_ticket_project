"""model/preprocessing.py
Data cleaning, train/test split, and multilingual feature union builder.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from model.config import DATA_PATH, CLEANED_DATA_PATH


def clean_data(filepath=DATA_PATH) -> pd.DataFrame:
    """Read CSV, fill missing subject/body, combine into 'text',
    normalize 'category' from 'queue', capitalize 'priority',
    fix business_type typos, dedupe, save cleaned CSV.
    Return the cleaned DataFrame."""
    df = pd.read_csv(filepath, encoding='latin1')

    # Fill missing text fields and create combined text
    df['subject'] = df['subject'].fillna('No Subject').astype(str).str.strip()
    df['body'] = df['body'].fillna('').astype(str).str.strip()
    df['text'] = (df['subject'] + " " + df['body']).str.strip()

    # Normalize category and priority
    cat_col = 'queue' if 'queue' in df.columns else 'category'
    df['category'] = df[cat_col].astype(str).str.strip()
    df['priority'] = df['priority'].astype(str).str.strip().str.capitalize()

    # Standardize business_type typos
    typo_map = {
        'Pit Services': 'IT Services',
        '_IT_Services_': 'IT Services',
        'IT Consulting Service': 'IT Consulting Firm'
    }
    df['business_type'] = df['business_type'].replace(typo_map)

    # Filter short texts, drop duplicates, and save
    df = df[df['text'].str.len() > 5].drop_duplicates(subset=['text']).reset_index(drop=True)
    df.to_csv(CLEANED_DATA_PATH, index=False, encoding='utf-8')
    return df


def create_text_feature_union(max_word_features=30000, max_char_features=15000) -> FeatureUnion:
    """Return FeatureUnion of word TF-IDF (1-3 grams) and 
    char_wb TF-IDF (3-5 grams), both with sublinear_tf=True."""
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
    """Stratified 80/20 split on 'category'.
    Return X_train, X_test, y_cat_train, y_cat_test, 
           y_pri_train, y_pri_test."""
    X = df['text']
    y_cat = df['category']
    y_pri = df['priority']

    return train_test_split(
        X, y_cat, y_pri,
        test_size=0.20,
        random_state=42,
        stratify=y_cat
    )
