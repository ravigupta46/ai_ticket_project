# ResolvAI — Intelligent Customer Support Ticket Triage Platform

ResolvAI is an end-to-end Machine Learning and Flask web platform designed to automatically triage, categorize, and prioritize customer support complaints. It leverages pure supervised machine learning pipelines (FeatureUnion of word and character n-grams + LinearSVC & ExtraTrees) paired with spaCy entity extraction to streamline support workflows.

---

## 📌 Architecture & Design

ResolvAI employs a clean, pure-ML inference architecture:
1. **Supervised ML Classification**: Incoming ticket subject and description are preprocessed and vectorized using a multilingual TF-IDF FeatureUnion (word 1-3 grams + character 3-5 grams), predicting both **category** and **priority** with high accuracy.
2. **Named Entity Recognition (NER)**: spaCy extracts critical domain entities such as organizations, amounts, dates, and order/reference numbers.
3. **Human-in-the-Loop Preview**: Users submit natural-language issues, review AI-generated titles, categories, priorities, and extracted entities in `preview_ticket.html`, and confirm before tickets are persisted.
4. **Security & State Management**: CSRF protection on all POST forms and AJAX status calls, hashed passwords via Werkzeug, secure session cookies (30-min lifetime), and server-side recomputation of entities on ticket confirmation.

---

## 🏷️ Canonical Taxonomy

**Categories (Queues):**
1. Billing and Payments
2. Customer Service
3. General Inquiry
4. Human Resources
5. IT Support
6. Product Support
7. Returns and Exchanges
8. Sales and Pre-Sales
9. Service Outages and Maintenance
10. Technical Support
11. Others

**Priorities:**
- High
- Medium
- Low

---

## 🤖 Automated Model Selection & ML Architecture

ResolvAI does not hardcode static classifiers. Instead, it utilizes an automated tournament-style model selection pipeline that trains and benchmarks multiple candidate architectures across stratified splits, performs hyperparameter grid tuning with cross-validation, and dynamically selects and persists whichever model achieves the highest validation performance (`f1_weighted`).

### 1. Feature Engineering: Multilingual Hybrid FeatureUnion
Before model benchmarking, incoming raw text (`subject` + `body`) is converted into semantic and syntactic representations using a dual-branch `FeatureUnion`:
* **Word n-grams (`ngram_range=(1, 3)`, max 30,000 features)**: Captures domain vocabulary, technical terminology, and phrase-level context.
* **Character n-grams (`analyzer='char_wb'`, `ngram_range=(3, 5)`, max 15,000 features)**: Provides resilience against misspellings, user typos, formatting quirks, and out-of-vocabulary words.
* **Sublinear TF Scaling (`sublinear_tf=True`)**: Dampens the effect of repetitive words by replacing TF with `1 + log(TF)`.

---

### 2. Candidate Model Evaluation & Dynamic Champion Selection

For each task (Category triage and Priority assessment), the training pipeline evaluates three distinct candidate model families. Each model undergoes cross-validated hyperparameter search (`GridSearchCV(cv=3, scoring='f1_weighted')`). The training pipeline automatically compares cross-validation scores and selects the top-performing candidate as the champion pipeline for deployment.

#### A. Category Classification Tournament
For routing tickets across 11 departmental queues, the pipeline evaluates three models suited for multi-class, high-dimensional text feature spaces:

| Candidate Model Family | Architecture Characteristics | Evaluated Hyperparameters | Selection Mechanism |
|---|---|---|---|
| **Multinomial Naive Bayes (`MultinomialNB`)** | Fast, probabilistic bag-of-words baseline | `classifier__alpha`: `[0.01, 0.1, 0.5, 1.0]` | Benchmark & cross-validate |
| **Logistic Regression (`LogisticRegression`)** | Linear boundary with calibrated probabilities & L2 regularization | `classifier__C`: `[0.5, 1.0, 3.0, 5.0, 10.0]` | Benchmark & cross-validate |
| **Linear Support Vector Classifier (`LinearSVC`)** | Large-margin boundary classifier optimized for sparse text representations | `classifier__C`: `[0.5, 0.8, 1.0, 1.2]` | Benchmark & cross-validate |

* **Selection Strategy**: The training runner dynamically tracks the highest weighted F1 score across all three candidate grids and saves the winning estimator directly to `model/category_pipeline.pkl`. In benchmark evaluations, `LinearSVC` achieved the highest score and was automatically selected.

#### B. Priority Classification Tournament
For estimating ticket urgency (`High`, `Medium`, `Low`), the pipeline evaluates linear and ensemble tree-based models capable of capturing non-linear urgency cues:

| Candidate Model Family | Architecture Characteristics | Evaluated Hyperparameters | Selection Mechanism |
|---|---|---|---|
| **Logistic Regression (`LogisticRegression`)** | Fast linear decision boundary across weighted features | `classifier__C`: `[0.5, 1.0, 3.0, 5.0]` | Benchmark & cross-validate |
| **Random Forest (`RandomForestClassifier`)** | Bagged ensemble of decision trees with feature subsampling | `classifier__n_estimators`: `[100, 150, 200]`, `classifier__min_samples_split`: `[2, 4]` | Benchmark & cross-validate |
| **Extra Trees Classifier (`ExtraTreesClassifier`)** | Extremely randomized decision trees minimizing split variance on sparse n-gram inputs | `classifier__n_estimators`: `[100, 150, 200]`, `classifier__min_samples_split`: `[2, 4]` | Benchmark & cross-validate |

* **Selection Strategy**: The training engine automatically compares validation performance across the three candidates and selects the estimator with the highest weighted F1 score, saving it to `model/priority_pipeline.pkl`. In benchmark evaluations, `ExtraTreesClassifier` outperformed the others and was automatically selected.

---

### 3. Named Entity Recognition (NER)
* **spaCy (`en_core_web_sm`)**: Selected for fast inference latency and low memory footprint in web request cycles.
* **Extracted Entities**:
  * Organizations & Systems (`ORG`)
  * Monetary & Transaction Amounts (`MONEY`)
  * Dates & Timelines (`DATE`)
  * Custom regex extractors for ticket/order references and transaction IDs.

---

## 📊 Evaluation Metrics (`model/output/metrics.json`)

All accuracy and F1 figures are evaluated strictly on the human-labelled, held-out test set (800 tickets, stratified 80/20 split from `new_Customer_support_data.csv`):

| Target | Champion Model | Accuracy | Weighted F1 | Test Evaluation |
|--------|----------------|----------|-------------|-----------------|
| **Category** | Linear Support Vector Classifier (LinearSVC) | **58.50%** | **0.5681** | Evaluated on 800 held-out customer tickets |
| **Priority** | Extra Trees Classifier (ExtraTrees) | **66.75%** | **0.6569** | Evaluated on 800 held-out customer tickets |

*Both models are evaluated on the ground-truth human-labelled held-out test set.*

---

## 📁 Repository Structure

```text
ai_ticket_project/
├── app.py                     # Flask web application (routes, CSRF, auth, session handling)
├── database.py                # SQLite persistence, schema setup, password hashing
├── create_agent.py            # CLI agent account provisioning utility
├── reset_db.py                # Database reset utility
├── requirements.txt           # Minimal runtime dependencies
├── build.sh                   # Deployment build script (spaCy download)
├── render.yaml                # Render cloud deployment specification
├── data/
│   └── new_Customer_support_data.csv  # Human-labelled support dataset
├── ai_engine/
│   ├── config.py              # Single source of truth for categories & priorities
│   └── ticket_generator.py    # Pure-ML ticket preview generator & entity extraction (singleton)
├── model/
│   ├── config.py              # Central ML paths, labels, and directory constants
│   ├── preprocessing.py       # Data cleaning, train/test split, FeatureUnion builder
│   ├── eda.py                 # Generates 5 high-resolution EDA plots
│   ├── train_category.py      # Category model comparison, GridSearchCV tuning, and export
│   ├── train_priority.py      # Priority model comparison, GridSearchCV tuning, and export
│   ├── evaluate.py            # Test set evaluation and metrics.json generation
│   ├── predict.py             # Single ticket prediction function and demo
│   ├── run_training.py        # Master entry point to execute the full pipeline
│   ├── category_pipeline.pkl  # Pickled category model pipeline
│   ├── priority_pipeline.pkl  # Pickled priority model pipeline
│   └── output/
│       ├── metrics.json       # Canonical evaluation metrics
│       ├── cleaned_customer_support_data.csv
│       └── eda_plots/         # Univariate and bivariate visualization plots
├── templates/                 # Jinja2 HTML templates
├── static/                    # CSS stylesheets and client-side JavaScript
└── tests/
    ├── test_app.py            # Flask route and CSRF tests
    ├── test_engine.py         # Unit tests for preprocessing, prediction, and singleton
    └── test_register.py       # User registration and security tests
```

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone <repo-url>
cd ai_ticket_project

# Install dependencies
pip install -r requirements.txt

# Download spaCy English NER model
python -m spacy download en_core_web_sm
```

### 2. Run Training Pipeline (Offline)
Run the full modular training, EDA, and evaluation flow in one command:
```bash
python model/run_training.py
```

Or test ticket inference directly:
```bash
python model/predict.py
```

### 3. Initialize Database & Provision Agent
```bash
# Initialize fresh database schema
python reset_db.py

# Provision an agent account via CLI
python create_agent.py <username>
```

### 4. Run Web Application
```bash
# Local development run
python app.py

# Production run with Gunicorn
gunicorn app:app
```
Access the application at `http://127.0.0.1:5000`.

---

## 🔒 Security & Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SECRET_KEY` | Random in prod / dev fallback | Flask session cryptographic signature |
| `FLASK_DEBUG` | `0` (False) | Enables debug mode (never enable in production) |
| `HOST` | `127.0.0.1` | Network binding interface |
| `PORT` | `5000` | Port for the HTTP server |
