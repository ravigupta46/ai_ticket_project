"""model/config.py
Single source of truth for paths and constants in the ML pipeline.
"""
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths: checks data/ directory or model/data/
DATA_PATH = os.path.join(BASE_DIR, "data", "new_Customer_support_data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "model", "data", "new_Customer_support_data.csv")

# Output and Plot directories
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "output")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda_plots")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_customer_support_data.csv")

# Model checkpoints and metrics
CAT_MODEL_PATH = os.path.join(BASE_DIR, "model", "category_pipeline.pkl")
PRI_MODEL_PATH = os.path.join(BASE_DIR, "model", "priority_pipeline.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# Taxonomy labels
CATEGORY_LABELS = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
    "Others"
]

PRIORITY_LABELS = ["Low", "Medium", "High"]

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
