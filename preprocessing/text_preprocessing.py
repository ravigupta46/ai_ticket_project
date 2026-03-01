# preprocessing/text_preprocessing.py (COMPLETELY REVISED)
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import os

print("="*60)
print("📝 TEXT PREPROCESSING")
print("="*60)

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")

print(f"\n📂 Input file: {INPUT_PATH}")
print(f"📂 Output directory: {OUTPUT_DIR}")

# Check if input file exists
if not os.path.exists(INPUT_PATH):
    print(f"❌ ERROR: Input file not found at {INPUT_PATH}")
    
    # Try alternative paths
    alt_paths = [
        os.path.join(BASE_DIR, "tickets.csv"),
        os.path.join(BASE_DIR, "data", "Consumer_Complaints.csv"),
        os.path.join(BASE_DIR, "Consumer_Complaints.csv")
    ]
    
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            INPUT_PATH = alt_path
            print(f"✅ Found input file at: {INPUT_PATH}")
            break
    else:
        print("❌ Could not find input file!")
        exit(1)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# First, let's see what columns are in the file
print("\n🔍 Examining file structure...")
try:
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df_header = None
    
    for encoding in encodings:
        try:
            df_header = pd.read_csv(INPUT_PATH, nrows=0, encoding=encoding)
            print(f"✅ Using encoding: {encoding}")
            break
        except:
            continue
    
    if df_header is None:
        df_header = pd.read_csv(INPUT_PATH, nrows=0, encoding='latin-1')
        print("⚠️ Using latin-1 encoding")
    
    print(f"\n📊 Columns found in dataset ({len(df_header.columns)} total):")
    for i, col in enumerate(df_header.columns):
        print(f"  {i+1}. '{col}'")
    
    # Now load actual data
    print("\n📥 Loading data (this may take a while for 300k+ rows)...")
    
    # Load in chunks to handle large file
    chunk_size = 50000
    chunks = []
    
    with tqdm(desc="Loading chunks") as pbar:
        for chunk in pd.read_csv(INPUT_PATH, chunksize=chunk_size, encoding=encoding, low_memory=False):
            chunks.append(chunk)
            pbar.update(1)
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded {len(df)} total rows")
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# Find the complaint narrative column
narrative_col = None
for col in df.columns:
    col_lower = col.lower()
    if any(word in col_lower for word in ['narrative', 'complaint', 'consumer complaint']):
        narrative_col = col
        print(f"\n✅ Found narrative column: '{narrative_col}'")
        break

if narrative_col is None:
    # Try to find any column with long text
    print("\n🔍 Looking for text column...")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check sample length
            sample = df[col].iloc[0] if len(df) > 0 else ""
            if isinstance(sample, str) and len(sample) > 100:
                narrative_col = col
                print(f"✅ Using column '{col}' as narrative (sample length: {len(sample)})")
                break

if narrative_col is None:
    print("❌ Could not find narrative column!")
    exit(1)

# Drop rows without narrative
initial_count = len(df)
df = df[df[narrative_col].notna()].copy()
print(f"\n📊 Rows with narrative: {len(df)} (dropped {initial_count - len(df)} rows)")

# Text cleaning function
def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove excessive newlines
    text = re.sub(r'[\r\n]+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\-\_\$\%]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n🧹 Cleaning text data...")
tqdm.pandas()
df["clean_text"] = df[narrative_col].progress_apply(clean_text)

# Remove very short texts (likely not useful)
initial_count = len(df)
df = df[df["clean_text"].str.len() > 20].copy()
print(f"📊 After removing short texts: {len(df)} rows (removed {initial_count - len(df)} rows)")

# Create category labels
print("\n🏷️ Creating category labels...")

def create_category(text):
    """Create category based on text content"""
    text = text.lower()
    
    # Define category keywords
    categories = {
        'Credit Card': ['credit card', 'creditcard', 'visa', 'mastercard', 'amex', 'discover'],
        'Banking': ['bank', 'checking', 'savings', 'account', 'deposit', 'withdraw', 'overdraft'],
        'Loan': ['loan', 'mortgage', 'refinance', 'borrow', 'lender'],
        'Debt Collection': ['debt', 'collection', 'collector', 'owe', 'collect'],
        'Student Loan': ['student', 'college', 'tuition', 'financial aid'],
        'Vehicle Loan': ['auto', 'car', 'vehicle', 'truck', 'lease'],
        'Payday Loan': ['payday', 'cash advance'],
        'Other': []
    }
    
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    
    return 'Other'

# Apply category creation
df['category'] = df['clean_text'].progress_apply(create_category)

# Show category distribution
print("\n📊 Category distribution:")
cat_counts = df['category'].value_counts()
for cat, count in cat_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {cat}: {count} ({percentage:.1f}%)")

# Create priority labels
print("\n⚠️ Creating priority labels...")

def create_priority(text):
    """Create priority based on text content"""
    text = text.lower()
    
    # High priority indicators
    high_keywords = [
        'urgent', 'asap', 'emergency', 'critical', 'immediately',
        'fraud', 'stolen', 'unauthorized', 'error', 'wrong',
        'legal', 'lawsuit', 'court', 'attorney'
    ]
    
    # Medium priority indicators
    medium_keywords = [
        'problem', 'issue', 'difficulty', 'trouble', 'confusing',
        'help', 'assist', 'confused', 'unclear', 'delay'
    ]
    
    if any(word in text for word in high_keywords):
        return 'High'
    elif any(word in text for word in medium_keywords):
        return 'Medium'
    else:
        return 'Low'

df['priority'] = df['clean_text'].progress_apply(create_priority)

# Show priority distribution
print("\n📊 Priority distribution:")
priority_counts = df['priority'].value_counts()
for pri, count in priority_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {pri}: {count} ({percentage:.1f}%)")

# Save processed data
print("\n💾 Saving processed data...")

# Save full dataset
full_path = os.path.join(OUTPUT_DIR, "cleaned_tickets.csv")
df.to_csv(full_path, index=False)
print(f"✅ Saved full dataset to: {full_path}")

# Save training data (just what we need for models)
training_path = os.path.join(OUTPUT_DIR, "training_data.csv")
df[["clean_text", "category", "priority"]].to_csv(training_path, index=False)
print(f"✅ Saved training data to: {training_path}")

print("\n" + "="*60)
print(f"✅ PREPROCESSING COMPLETED!")
print(f"📊 Final dataset size: {len(df)} rows")
print("="*60)

# Show sample
print("\n📝 Sample of processed data:")
print(df[["clean_text", "category", "priority"]].head(3))