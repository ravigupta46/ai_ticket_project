# process_large_dataset.py (UPDATED)
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import chardet

def detect_encoding(filepath):
    """Detect the encoding of the file"""
    print("Detecting file encoding...")
    with open(filepath, 'rb') as f:
        raw_data = f.read(100000)  # Read first 100KB for detection
        result = chardet.detect(raw_data)
        print(f"Detected encoding: {result['encoding']} with confidence: {result['confidence']}")
        return result['encoding']

def process_in_chunks(filepath, chunk_size=10000):
    """Process large CSV file in chunks with proper encoding"""
    
    print(f"Processing {filepath} in chunks of {chunk_size} rows...")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ Error: File {filepath} not found!")
        return None
    
    # Detect encoding
    try:
        encoding = detect_encoding(filepath)
    except:
        # Fallback encodings to try
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        encoding = None
        for enc in encodings:
            try:
                print(f"Trying encoding: {enc}")
                # Test if we can read the first few lines
                with open(filepath, 'r', encoding=enc) as f:
                    f.readline()
                encoding = enc
                print(f"✅ Success with encoding: {enc}")
                break
            except:
                continue
        
        if encoding is None:
            print("❌ Could not find suitable encoding. Using 'latin-1' as fallback.")
            encoding = 'latin-1'
    
    # Count rows efficiently without loading full file
    try:
        print("Counting total rows...")
        total_rows = 0
        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            for _ in f:
                total_rows += 1
        total_rows -= 1  # Subtract header
        print(f"Total rows in file: {total_rows:,}")
    except Exception as e:
        print(f"Could not count rows: {e}")
        total_rows = None
    
    chunks = []
    try:
        with tqdm(total=total_rows, desc="Processing chunks", unit="rows") as pbar:
            for i, chunk in enumerate(pd.read_csv(
                filepath, 
                chunksize=chunk_size, 
                encoding=encoding,
                encoding_errors='ignore',  # Ignore problematic characters
                low_memory=False,
                on_bad_lines='warn'  # Warn about bad lines but continue
            )):
                # Process chunk
                processed = process_chunk(chunk, i)
                if processed is not None and len(processed) > 0:
                    chunks.append(processed)
                
                if total_rows:
                    pbar.update(len(chunk))
                else:
                    pbar.update(chunk_size)
                    
    except Exception as e:
        print(f"Error while reading CSV: {e}")
        print("Trying with different encoding...")
        
        # Fallback: use latin-1 which accepts all byte values
        print("Using latin-1 encoding as fallback...")
        with tqdm(desc="Processing chunks (fallback)", unit="rows") as pbar:
            for i, chunk in enumerate(pd.read_csv(
                filepath, 
                chunksize=chunk_size, 
                encoding='latin-1',
                low_memory=False,
                on_bad_lines='skip'  # Skip bad lines
            )):
                processed = process_chunk(chunk, i)
                if processed is not None and len(processed) > 0:
                    chunks.append(processed)
                pbar.update(len(chunk))
    
    # Combine all chunks
    if chunks:
        final_df = pd.concat(chunks, ignore_index=True)
        print(f"\n✅ Processed {len(final_df):,} rows total")
        return final_df
    else:
        print("❌ No data processed")
        return None

def process_chunk(chunk, chunk_num):
    """Process a single chunk of data"""
    
    # Display first chunk's columns for debugging
    if chunk_num == 0:
        print("\n📊 Available columns in dataset:")
        for i, col in enumerate(chunk.columns[:20]):  # Show first 20 columns
            print(f"  {i+1}. {col}")
        
        # Try to find text columns
        text_cols = []
        for col in chunk.columns:
            if chunk[col].dtype == 'object':  # Text columns are usually object type
                # Check if it contains long text
                sample = str(chunk[col].iloc[0]) if len(chunk) > 0 else ""
                if len(sample) > 50:  # Likely a narrative column
                    text_cols.append((col, len(sample)))
        
        if text_cols:
            print("\n📝 Potential narrative columns:")
            for col, length in sorted(text_cols, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {col} (sample length: {length})")
    
    # Look for complaint narrative column
    narrative_col = None
    for col in chunk.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['narrative', 'complaint', 'description', 'text', 'story', 'issue']):
            narrative_col = col
            if chunk_num == 0:
                print(f"\n✅ Found narrative column: {narrative_col}")
            break
    
    if narrative_col is None:
        # Fallback: use first text column with longest average length
        text_cols = chunk.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            # Find column with longest average text length
            avg_lengths = {}
            for col in text_cols[:10]:  # Check first 10 text columns
                try:
                    avg_lengths[col] = chunk[col].astype(str).str.len().mean()
                except:
                    pass
            
            if avg_lengths:
                narrative_col = max(avg_lengths, key=avg_lengths.get)
                if chunk_num == 0:
                    print(f"\n⚠️ Using fallback column: {narrative_col}")
    
    if narrative_col:
        # Keep only rows with non-null values
        chunk = chunk[chunk[narrative_col].notna()].copy()
        
        # Convert to string and clean basic
        chunk['clean_text'] = chunk[narrative_col].astype(str).apply(
            lambda x: x[:5000] if len(x) > 5000 else x  # Limit length
        )
        
        return chunk[['clean_text']]
    else:
        if chunk_num == 0:
            print("❌ No suitable text column found!")
        return None

def create_training_data():
    """Main function to create training data from large dataset"""
    
    print("="*60)
    print("🚀 LARGE DATASET PROCESSOR")
    print("="*60)
    
    # Process the large dataset
    filepath = "data/tickets.csv"
    df = process_in_chunks(filepath, chunk_size=50000)  # Process 50k rows at a time
    
    if df is None or len(df) == 0:
        print("❌ No data to process")
        return
    
    # Remove duplicates and very short texts
    initial_count = len(df)
    df = df.drop_duplicates(subset=['clean_text'])
    df = df[df['clean_text'].str.len() > 20]
    
    print(f"\n📊 Dataset statistics:")
    print(f"  - Initial rows: {initial_count:,}")
    print(f"  - After cleaning: {len(df):,}")
    print(f"  - Removed: {initial_count - len(df):,} rows")
    
    # Show samples
    print("\n📝 Sample entries:")
    for i, text in enumerate(df['clean_text'].head(3)):
        print(f"\n  {i+1}. {text[:200]}..." if len(text) > 200 else f"\n  {i+1}. {text}")
    
    # Save to output folder
    os.makedirs('model/output', exist_ok=True)
    output_file = 'model/output/cleaned_tickets.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Saved cleaned data to: {output_file}")
    
    return df

if __name__ == "__main__":
    # Install chardet if not available
    try:
        import chardet
    except ImportError:
        print("Installing chardet for encoding detection...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'chardet'])
        import chardet
    
    create_training_data()