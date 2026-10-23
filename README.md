# ResolvAI — Intelligent Customer Support Ticket Triage Platform

ResolvAI is an end-to-end Machine Learning and Flask web platform designed to automatically triage, categorize, and prioritize customer support complaints. It combines deterministic word-boundary regex rules, supervised machine learning pipelines, and entity extraction to streamline support workflows.

---

## 📌 Architecture & Design

ResolvAI employs a **hybrid inference** architecture:
1. **Deterministic Rule Engine First**: High-precision, compiled word-boundary regular expressions evaluate queries against domain signatures. Rule 12 (`Product Queries`) strictly requires both hardware symptoms and device keywords so it does not swallow general inquiries.
2. **Supervised ML Fallback**: If rules return `Others`, queries are evaluated by an end-to-end `sklearn.pipeline.Pipeline` (TF-IDF Vectorizer + Classifier) trained on customer support remarks.
3. **Human-in-the-Loop Preview**: Users submit natural-language issues, review AI-generated titles, categories, priorities, and extracted entities in `preview_ticket.html`, and confirm before tickets are persisted.
4. **Security & State Management**: CSRF protection on all POST forms and AJAX status calls, hashed passwords via Werkzeug, secure session cookies (30-min lifetime), and server-side recomputation of entities on ticket confirmation.

---

## 🏷️ Canonical Taxonomy (12 Categories)

1. **Product Queries** (Hardware issues, specifications, warranty)
2. **Returns** (Damaged product, reverse pickup, replacement)
3. **Order Related** (Tracking, delivery delays, invoice requests)
4. **Refund Related** (Bank credit, COD refunds, refund status)
5. **Payments related** (Gateway errors, double debits, wallet/EMI)
6. **Cancellation** (Order cancellation, stopping delivery)
7. **Shopzilla Related** (Account access, login, OTP, profile)
8. **Feedback** (Agent demeanor, delivery feedback)
9. **Offers & Cashback** (Coupons, promo codes, wallet cashback)
10. **App/website** (App crashes, website loading bugs)
11. **Onboarding related** (Merchant/seller registration, commission)
12. **Others** (General inquiries, callback requests)

---

## 📊 Honest Evaluation Metrics (`model/output/metrics.json`)

All accuracy figures in the project are derived strictly from `model/output/metrics.json`. The test set consists of real, held-out, deduplicated customer remarks (no synthetic or domain-phrase leakage in test):

| Metric Component | Model / Method | Accuracy | Weighted F1 | Notes |
|------------------|----------------|----------|-------------|-------|
| **Category (ML-Only)** | Multinomial Naive Bayes | **54.34%** | **0.4587** | Evaluated on 3,344 real held-out customer remarks |
| **Category (Hybrid)** | Regex Rules + ML Fallback | **52.15%** | **0.4572** | Rules resolve 14.65% of test queries; 85.35% fall back to ML |
| **Priority Model** | Random Forest Pipeline | **94.81%** | **0.9466** | **Weak-label (rule-derived), not human-verified** |

> **Note on Priority Labels:** Ground-truth human annotations were not available in the dataset; priority labels are derived from shared urgency keywords (weak supervision). A documented hook `load_priority_dataset(custom_path)` is provided in `model/priority_model.py` to seamlessly ingest human-verified labels.

---

## 📁 Repository Structure

```text
ai_ticket_project/
├── app.py                     # Flask application (CSRF, sessions, validation, routes)
├── database.py                # SQLite persistence, migrations, password hashing
├── create_agent.py            # CLI agent account provisioning utility
├── reset_db.py                # Database reset utility
├── requirements.txt           # Minimal, clean runtime dependencies
├── build.sh                   # Deployment build script (spaCy download)
├── render.yaml                # Render cloud deployment specification
├── ai_engine/
│   ├── config.py              # Single source of truth for categories, priorities & compiled regex
│   ├── rules.py               # Deterministic regex rule evaluation
│   └── ticket_generator.py    # Hybrid inference engine & entity extractor (singleton)
├── model/
│   ├── data/
│   │   └── domain_phrases.json  # Hand-curated domain phrases for training oversampling
│   ├── train_category_model.py  # Stratified training pipeline & metrics generator
│   ├── priority_model.py        # Weak-supervision priority classifier pipeline
│   ├── topic_modelling_nmf.py   # Unsupervised NMF topic analysis
│   ├── eda.py                   # Exploratory data analysis
│   └── output/
│       ├── metrics.json         # Single source of truth for all project metrics
│       ├── training_data.csv    # Curated training data
│       └── model_comparison_results.json
├── templates/                 # Jinja2 HTML templates
├── static/                    # CSS stylesheets and client-side JavaScript
└── tests/
    └── test_engine.py         # Pytest verification suite
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
```bash
# Train category classifier, select best model, and produce metrics.json
python model/train_category_model.py

# Train priority pipeline with weak labels
python model/priority_model.py

# Run optional unsupervised topic modeling & EDA
python model/topic_modelling_nmf.py
python model/eda.py
```

### 3. Initialize Database & Provision Agent
```bash
# Initialize fresh database schema
python reset_db.py

# Provision an agent account via CLI
python create_agent.py <username>
```
Agents are provisioned by an administrator via `python create_agent.py <username>`; public registration creates customer accounts only.

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
| `PRIORITY_DATA_PATH` | None | Optional path to human-verified priority dataset |