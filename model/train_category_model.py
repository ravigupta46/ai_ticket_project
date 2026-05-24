# model/train_category_model.py (UPDATED - Clear accuracy display)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*80)
print("🚀 CATEGORY MODEL TRAINING")
print("="*80)
print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Get the correct paths based on your folder structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'model', 'output', 'training_data.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # model folder
OUTPUT_DIR = os.path.join(BASE_DIR, 'model', 'output')

print(f"\n📂 Looking for training data at: {DATA_PATH}")

# Check if file exists
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: Training data not found at {DATA_PATH}")
    
    # Try alternative locations
    alt_paths = [
        os.path.join(BASE_DIR, 'model', 'output', 'cleaned_tickets.csv'),
        os.path.join(BASE_DIR, 'output', 'training_data.csv'),
        os.path.join(BASE_DIR, 'data', 'training_data.csv'),
    ]
    
    print("\n🔍 Searching alternative locations...")
    found = False
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            DATA_PATH = alt_path
            print(f"✅ Found at: {DATA_PATH}")
            found = True
            break
    
    if not found:
        print("\n❌ Could not find training data file!")
        print("\nPlease run preprocessing first:")
        print("  python preprocessing/text_preprocessing.py")
        exit(1)

# Load the data
print(f"\n📥 Loading training data...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} rows")
    print(f"📊 Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# Check if we have the required columns
if 'category' not in df.columns:
    print("\n⚠️ Category column not found. Creating from text...")
    
    def determine_category(text):
        text = str(text).lower()
        if any(word in text for word in ['credit card', 'visa', 'mastercard', 'amex', 'discover']):
            return 'Credit Card'
        elif any(word in text for word in ['bank', 'checking', 'savings', 'account', 'deposit', 'withdraw']):
            return 'Banking'
        elif any(word in text for word in ['loan', 'mortgage', 'refinance', 'borrow']):
            return 'Loan'
        elif any(word in text for word in ['debt', 'collection', 'collector', 'owe']):
            return 'Debt Collection'
        elif any(word in text for word in ['student', 'college', 'tuition', 'financial aid']):
            return 'Student Loan'
        else:
            return 'Other'
    
    if 'clean_text' in df.columns:
        df['category'] = df['clean_text'].apply(determine_category)
    else:
        # Try to find a text column
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            df['clean_text'] = df[text_cols[0]].astype(str)
            df['category'] = df['clean_text'].apply(determine_category)
        else:
            print("❌ No text column found for category creation!")
            exit(1)

# Check if we have clean_text column
if 'clean_text' not in df.columns:
    print("\n⚠️ clean_text column not found. Looking for text column...")
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        df['clean_text'] = df[text_cols[0]].astype(str)
        print(f"✅ Using '{text_cols[0]}' as text column")
    else:
        print("❌ No text column found!")
        exit(1)

# Remove any rows with missing values
initial_len = len(df)
df = df.dropna(subset=['clean_text', 'category'])
df = df[df['clean_text'].str.len() > 10]
print(f"\n📊 After cleaning: {len(df)} rows (removed {initial_len - len(df)} rows)")

# Show category distribution
print("\n" + "-"*80)
print("📊 CATEGORY DISTRIBUTION")
print("-"*80)
category_counts = df['category'].value_counts()
total = len(df)
for cat, count in category_counts.items():
    percentage = (count / total) * 100
    bar = "█" * int(percentage/2) + "░" * (50 - int(percentage/2))
    print(f"  {cat:15} : {count:6} ({percentage:5.1f}%) {bar}")

# Prepare features and labels
X = df["clean_text"].fillna("")
y = df["category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {len(X_train)} samples")
print(f"📊 Test set: {len(X_test)} samples")

# Vectorize text
print("\n🔧 Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

print("Vectorizing training text...")
X_train_vec = vectorizer.fit_transform(X_train)
print(f"Vectorized shape: {X_train_vec.shape}")

print("Vectorizing test text...")
X_test_vec = vectorizer.transform(X_test)

# Train model
print("\n🤖 Training category model...")
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

# Evaluate
print("\n" + "="*80)
print("📊 MODEL EVALUATION RESULTS")
print("="*80)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy prominently (WILL SHOW CLEARLY)
print("\n" + "█"*80)
print("█"*80)
print(f"███      CATEGORY MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)      ███")
print("█"*80)
print("█"*80 + "\n")

# Show per-class accuracy
print("\n" + "-"*80)
print("📊 PER-CLASS PERFORMANCE")
print("-"*80)
report = classification_report(y_test, y_pred, output_dict=True)
for label in report.keys():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"\n  {label}:")
        print(f"    Precision: {report[label]['precision']:.4f}")
        print(f"    Recall:    {report[label]['recall']:.4f}")
        print(f"    F1-Score:  {report[label]['f1-score']:.4f}")

# Cross-validation
print("\n" + "-"*80)
print("📊 CROSS-VALIDATION SCORES")
print("-"*80)
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
print(f"  CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Save model
print("\n💾 Saving model...")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'category_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved to: {model_path}")

# Save vectorizer
vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✅ Vectorizer saved to: {vectorizer_path}")

# Save accuracy for dashboard
os.makedirs(OUTPUT_DIR, exist_ok=True)
accuracy_path = os.path.join(OUTPUT_DIR, 'category_accuracy.txt')
with open(accuracy_path, 'w') as f:
    f.write(str(accuracy))
print(f"✅ Accuracy saved to: {accuracy_path}")

print("\n" + "="*80)
print("✅ CATEGORY MODEL TRAINING COMPLETED SUCCESSFULLY!")
print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# FINAL ACCURACY DISPLAY - VERY CLEAR
print("\n" + "🔥"*40)
print("🔥"*40)
print(f"🔥           FINAL CATEGORY ACCURACY: {accuracy*100:.2f}%                🔥")
print("🔥"*40)
print("🔥"*40 + "\n")