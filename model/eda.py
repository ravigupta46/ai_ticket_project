"""model/eda.py
Exploratory data analysis: generates 5 readable plots saved to EDA_DIR.
"""
import os
import tempfile
import pandas as pd

# Headless backend and safe temp cache
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model.config import EDA_DIR


def perform_eda(df: pd.DataFrame) -> None:
    """Generate 5 PNG plots:
       01_category_distribution.png
       02_priority_distribution.png
       03_text_length_distribution.png
       04_length_by_priority.png
       05_category_vs_priority.png
    Save to EDA_DIR. Do not return anything."""
    sns.set_theme(style="whitegrid", palette="muted")
    df['text_char_len'] = df['text'].fillna('').astype(str).str.len()
    pri_order = ['Low', 'Medium', 'High']
    pri_colors = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}

    # 1. Category Distribution
    plt.figure(figsize=(10, 5))
    cat_order = df['category'].value_counts().index
    sns.countplot(data=df, y='category', order=cat_order, hue='category', legend=False, palette='viridis')
    plt.title("Ticket Category (Queue) Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Ticket Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "01_category_distribution.png"), dpi=200)
    plt.close()

    # 2. Priority Distribution
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x='priority', order=pri_order, hue='priority', palette=pri_colors, legend=False)
    plt.title("Ticket Priority Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Priority Level")
    plt.ylabel("Ticket Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "02_priority_distribution.png"), dpi=200)
    plt.close()

    # 3. Text Length Distribution
    plt.figure(figsize=(9, 4.5))
    sns.histplot(df['text_char_len'], bins=40, kde=True, color='#3b82f6')
    plt.title("Ticket Text Character Length Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Character Length (Subject + Body)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "03_text_length_distribution.png"), dpi=200)
    plt.close()

    # 4. Text Length by Priority
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x='priority', y='text_char_len', order=pri_order, palette=pri_colors, showfliers=False)
    plt.title("Text Length vs Priority Level", fontsize=13, fontweight='bold')
    plt.xlabel("Priority")
    plt.ylabel("Character Length")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "04_length_by_priority.png"), dpi=200)
    plt.close()

    # 5. Category vs Priority Breakdown
    plt.figure(figsize=(11, 6))
    ct = pd.crosstab(df['category'], df['priority'], normalize='index')[pri_order] * 100
    ct.plot(kind='barh', stacked=True, color=[pri_colors[p] for p in pri_order], figsize=(10, 6))
    plt.title("Priority Breakdown across Ticket Categories (%)", fontsize=13, fontweight='bold')
    plt.xlabel("Percentage (%)")
    plt.ylabel("Category")
    plt.legend(title="Priority", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "05_category_vs_priority.png"), dpi=200)
    plt.close()
