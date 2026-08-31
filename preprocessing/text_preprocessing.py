# preprocessing/text_preprocessing.py
import os
import sys
import json
import re
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

print("="*70)
print("[STEP 1 & 2] DATA LOADING AND TEXT PREPROCESSING")
print("="*70)

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load spaCy model
print("\n[INFO] Loading spaCy English model...")
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
print("[OK] spaCy loaded successfully")

# -------------------------------------------------------------------
# 1. DATA LOADING
# -------------------------------------------------------------------
def load_complaints_data():
    """Load complaints data from JSON or CSV fallback"""
    # Look for JSON file first
    possible_json_files = [
        os.path.join(BASE_DIR, "complaints-2021-05-14_08_16.json"),
        os.path.join(DATA_DIR, "complaints-2021-05-14_08_16.json"),
        os.path.join(BASE_DIR, "complaints.json"),
        os.path.join(DATA_DIR, "complaints.json")
    ]
    
    for json_file in possible_json_files:
        if os.path.exists(json_file):
            print(f"[INFO] Found JSON dataset at: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            
            # Standardize column renaming
            rename_dict = {
                '_index': 'index',
                '_type': 'type',
                '_id': 'id',
                '_score': 'score',
                '_source.tags': 'tags',
                '_source.zip_code': 'zip_code',
                '_source.complaint_id': 'complaint_id',
                '_source.issue': 'issue',
                '_source.date_received': 'date_received',
                '_source.state': 'state',
                '_source.consumer_disputed': 'consumer_disputed',
                '_source.product': 'product',
                '_source.company_response': 'company_response',
                '_source.company': 'company',
                '_source.submitted_via': 'submitted_via',
                '_source.date_sent_to_company': 'date_sent_to_company',
                '_source.company_public_response': 'company_public_response',
                '_source.sub_product': 'sub_product',
                '_source.timely': 'timely',
                '_source.complaint_what_happened': 'complaint_what_happened',
                '_source.sub_issue': 'sub_issue',
                '_source.consumer_consent_provided': 'consumer_consent_provided'
            }
            df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)
            return df
            
    print("[WARNING] JSON complaint file not found. Checking CSV datasets...")
    possible_csv_files = [
        os.path.join(DATA_DIR, "Customer_support_data.csv"),
        os.path.join(DATA_DIR, "tickets.csv"),
        os.path.join(BASE_DIR, "tickets.csv")
    ]
    
    for csv_file in possible_csv_files:
        if os.path.exists(csv_file):
            print(f"[INFO] Loading CSV dataset at: {csv_file}")
            try:
                df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')
            except Exception:
                df = pd.read_csv(csv_file, low_memory=False, encoding='latin-1')
                
            # Map text column to complaint_what_happened
            text_col = None
            for col in ['Customer Remarks', 'complaint_what_happened', 'narrative', 'complaint', 'clean_text', 'description']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col:
                df.rename(columns={text_col: 'complaint_what_happened'}, inplace=True)
                if 'product' not in df.columns and 'category' in df.columns:
                    df['product'] = df['category']
                if 'sub_product' not in df.columns and 'Sub-category' in df.columns:
                    df['sub_product'] = df['Sub-category']
                return df
                
    # If no files found, generate starter financial complaint dataset
    print("[INFO] Creating sample financial complaints dataset for demonstration...")
    sample_data = {
        'complaint_what_happened': [
            "I cannot get from Chase who services my mortgage, who owns it and who has original loan docs. I need mortgage payoff statement.",
            "The bill amount of my credit card was debited twice. Please look into the matter and resolve at the earliest with refund.",
            "I want to open a salary account at your downtown branch. Please provide me the procedure and checking deposit requirements.",
            "Yesterday, I received a fraudulent email regarding renewal of my services. An unauthorized transaction of $500 occurred on my card.",
            "What is the procedure to know my CIBIL credit score and report dispute against late payment mark?",
            "I need to know the number of bank branches and their locations for account withdrawal and cashier check services.",
            "My credit card interest rate was increased without prior notice and late payment fee was charged wrongfully.",
            "The mortgage lender failed to escrow property tax correctly causing penalty from tax assessor.",
            "Identity theft incident where unknown person opened loan account in my name without verification.",
            "Unable to access online banking app to transfer money between savings and checking accounts.",
            "Card skimming theft reported at ATM machine. $1200 withdrawn without my authorization.",
            "Refinancing home loan paperwork delayed over 90 days causing interest rate lock expiration.",
            "Dispute regarding billing error on annual fee for rewards credit card.",
            "Customer service branch representative was very rude and unhelpful during account closure inquiry.",
            "Report unauthorized wire transfer from commercial business bank account to foreign bank."
        ] * 40,
        'product': ['Mortgage', 'Credit Card', 'Bank Account', 'Theft', 'Other'] * 120,
        'sub_product': ['Conventional', 'Cashback', 'Checking', 'Fraud', 'General'] * 120
    }
    return pd.DataFrame(sample_data)

# Load data
df = load_complaints_data()
print(f"[DATA] Total raw records: {len(df)}")

# Filter empty complaints
if 'complaint_what_happened' in df.columns:
    df['complaint_what_happened'].replace('', np.nan, inplace=True)
    df.dropna(subset=['complaint_what_happened'], inplace=True)
    df = df[df['complaint_what_happened'].astype(str).str.strip().str.len() > 10].copy()
    print(f"[DATA] Valid complaints with narrative: {len(df)}")
else:
    raise ValueError("[ERROR] Column 'complaint_what_happened' not found in dataset!")

# Limit processing sample size if dataset is extremely large for fast, responsive processing
MAX_SAMPLES = 25000
if len(df) > MAX_SAMPLES:
    print(f"[INFO] Sampling {MAX_SAMPLES} records for optimal topic modeling speed...")
    df = df.sample(n=MAX_SAMPLES, random_state=40).reset_index(drop=True)

# -------------------------------------------------------------------
# 2. TEXT CLEANING & PREPROCESSING
# -------------------------------------------------------------------
def clean_text(text):
    """
    Clean text:
    - Lowercase
    - Remove text in square brackets
    - Remove punctuation
    - Remove words with numbers
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)       # Remove text in square brackets
    text = re.sub(r'[^\w\s]', ' ', text)       # Remove punctuation
    text = re.sub(r'\w*\d\w*', ' ', text)      # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text).strip()   # Remove extra spaces
    return text

def lemmatization(texts):
    """Lemmatize text and remove stopwords using spaCy"""
    lemma_sentences = []
    for doc in tqdm(nlp.pipe(texts, batch_size=256, n_process=1), total=len(texts), desc="Lemmatizing"):
        sent = [token.lemma_ for token in doc if token.text not in stopwords and len(token.lemma_) > 1]
        lemma_sentences.append(' '.join(sent))
    return lemma_sentences

def extract_pos_tags(texts):
    """Extract only NN POS tags (nouns)"""
    pos_sentences = []
    for doc in tqdm(nlp.pipe(texts, batch_size=256, n_process=1), total=len(texts), desc="Extracting NN POS"):
        sent = [token.text for token in doc if token.tag_ == 'NN']
        pos_sentences.append(' '.join(sent))
    return pos_sentences

# Create df_clean
print("\n[CLEAN] 1. Applying basic text cleaning...")
tqdm.pandas(desc="Cleaning text")
df_clean = pd.DataFrame()
df_clean['complaint_what_happened'] = df['complaint_what_happened'].progress_apply(clean_text)

# Drop any that became empty after basic cleaning
df_clean = df_clean[df_clean['complaint_what_happened'].str.len() > 5].reset_index(drop=True)

print("\n[CLEAN] 2. Applying lemmatization & stopword removal...")
df_clean['complaint_what_happened_lemmatized'] = lemmatization(df_clean['complaint_what_happened'].tolist())

if 'product' in df.columns:
    df_clean['category'] = df['product'].values[:len(df_clean)]
if 'sub_product' in df.columns:
    df_clean['sub_category'] = df['sub_product'].values[:len(df_clean)]

print("\n[CLEAN] 3. Extracting NN POS tags...")
df_clean['complaint_POS_removed'] = extract_pos_tags(df_clean['complaint_what_happened_lemmatized'].tolist())

print("\n[CLEAN] 4. Removing -PRON- and masked personal details (xxxx)...")
df_clean['Complaint_clean'] = df_clean['complaint_POS_removed'].str.replace('-PRON-', '', regex=False)
df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.replace('xxxx', '', regex=False)
df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.replace('xx', '', regex=False)
df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Filter out empty clean complaints
df_clean = df_clean[df_clean['Complaint_clean'].str.len() > 3].reset_index(drop=True)

# -------------------------------------------------------------------
# SAVE PREPROCESSED DATA
# -------------------------------------------------------------------
output_clean_path = os.path.join(OUTPUT_DIR, "cleaned_tickets.csv")
df_clean.to_csv(output_clean_path, index=False, encoding='utf-8')
print(f"\n[SAVE] Saved preprocessed data to: {output_clean_path} ({len(df_clean)} records)")

print("\n" + "="*70)
print("[SUCCESS] PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nSample processed records:")
print(df_clean[['complaint_what_happened', 'Complaint_clean']].head(3))
