# ai_engine/ticket_generator.py
import os
import re
import pickle
import logging
import spacy
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

CATEGORY_MAPPING = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan',
    '0': 'Bank Account services',
    '1': 'Credit card or prepaid card',
    '2': 'Others',
    '3': 'Theft/Dispute Reporting',
    '4': 'Mortgage/Loan',
    'Bank Account services': 'Bank Account services',
    'Credit card or prepaid card': 'Credit card or prepaid card',
    'Theft/Dispute Reporting': 'Theft/Dispute Reporting',
    'Mortgage/Loan': 'Mortgage/Loan',
    'Others': 'Others'
}

class AITicketGenerator:
    def __init__(self):
        self.models_loaded = False
        self.category_model = None
        self.priority_model = None
        self.vectorizer = None
        self.priority_vectorizer = None
        self.nlp = None
        self.load_models()

    def load_models(self):
        """Load trained category and priority models & vectorizers"""
        try:
            category_model_file = os.path.join(MODEL_DIR, 'category_model.pkl')
            vectorizer_file = os.path.join(MODEL_DIR, 'vectorizer.pkl')
            priority_model_file = os.path.join(MODEL_DIR, 'priority_model.pkl')
            priority_vectorizer_file = os.path.join(MODEL_DIR, 'priority_vectorizer.pkl')

            # Load Category Model & Vectorizer
            if os.path.exists(category_model_file) and os.path.exists(vectorizer_file):
                with open(category_model_file, 'rb') as f:
                    self.category_model = pickle.load(f)
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("[OK] Category model & vectorizer loaded successfully")

            # Load Priority Model & Vectorizer
            if os.path.exists(priority_model_file) and os.path.exists(priority_vectorizer_file):
                with open(priority_model_file, 'rb') as f:
                    self.priority_model = pickle.load(f)
                with open(priority_vectorizer_file, 'rb') as f:
                    self.priority_vectorizer = pickle.load(f)
                logger.info("[OK] Priority model loaded successfully")

            # Load spaCy NLP Model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("[OK] spaCy NER model loaded")
            except Exception:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")

            if self.category_model is not None and self.vectorizer is not None:
                self.models_loaded = True
                logger.info("[OK] ML Ticket Generator initialized in ML mode")
            else:
                logger.warning("[WARN] ML models not found on disk. Operating in rule-based fallback mode")

        except Exception as e:
            logger.error(f"[ERROR] Error loading models: {e}")
            self.models_loaded = False

    def preprocess_text(self, text):
        """Clean and normalize input text"""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\w*\d\w*', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_entities(self, text):
        """Extract financial entities, organizations, money amounts, dates, and account/error codes"""
        entities = {
            'error_codes': [],
            'organizations': [],
            'amounts': [],
            'dates': [],
            'products': []
        }

        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ == 'MONEY':
                    entities['amounts'].append(ent.text)
                elif ent.label_ == 'PRODUCT':
                    entities['products'].append(ent.text)

        # Regex patterns for reference codes, transaction IDs, account numbers
        ref_patterns = [
            r'\b[A-Z0-9]{6,16}\b',
            r'\b\$\s?\d+(?:,\d{3})*(?:\.\d{2})?\b',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        ]
        for pattern in ref_patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                if '$' in m:
                    entities['amounts'].append(m)
                elif len(m) >= 6:
                    entities['error_codes'].append(m)

        # Deduplicate and limit
        for k in entities:
            entities[k] = list(dict.fromkeys(entities[k]))[:5]

        return entities

    def generate_title(self, text):
        """Generate a concise, professional title from complaint text"""
        sentences = re.split(r'[.!?\n]', text)
        first_sentence = sentences[0].strip() if sentences else text[:60]
        
        if len(first_sentence) > 75:
            title = first_sentence[:75].rsplit(' ', 1)[0] + '...'
        else:
            title = first_sentence

        title = title.strip().capitalize()
        if not title.endswith(('.', '!', '?', '...')):
            title += '...'
        return title

    def predict_category(self, text):
        """Predict one of the 5 financial complaint categories"""
        if self.models_loaded and self.category_model is not None and self.vectorizer is not None:
            try:
                vec = self.vectorizer.transform([text])
                pred = self.category_model.predict(vec)[0]
                category_name = CATEGORY_MAPPING.get(pred, str(pred))
                return category_name
            except Exception as e:
                logger.error(f"Inference error in category prediction: {e}")
                return self.rule_based_category(text)
        return self.rule_based_category(text)

    def predict_priority(self, text):
        """Predict Priority (High, Medium, Low)"""
        # Always check high urgency critical keywords first
        text_lower = text.lower()
        urgent_keywords = ['urgent', 'emergency', 'fraud', 'stolen', 'unauthorized', 'skimming', 'critical', 'immediately']
        if any(w in text_lower for w in urgent_keywords):
            return 'High'

        if self.priority_model is not None and self.priority_vectorizer is not None:
            try:
                vec = self.priority_vectorizer.transform([text])
                pred = self.priority_model.predict(vec)[0]
                return str(pred)
            except Exception as e:
                logger.error(f"Inference error in priority prediction: {e}")
                return self.rule_based_priority(text)
        return self.rule_based_priority(text)

    def rule_based_category(self, text):
        """Rule-based fallback for the 5 financial complaint categories"""
        t = text.lower()
        if any(w in t for w in ['mortgage', 'loan', 'escrow', 'refinance', 'payoff', 'foreclosure', 'lender', 'home loan', 'property tax']):
            return 'Mortgage/Loan'
        elif any(w in t for w in ['credit card', 'prepaid card', 'card limit', 'annual fee', 'late fee', 'visa', 'mastercard', 'amex', 'cashback']):
            return 'Credit card or prepaid card'
        elif any(w in t for w in ['fraud', 'theft', 'unauthorized', 'scam', 'stolen', 'skimming', 'dispute', 'identity theft', 'fake']):
            return 'Theft/Dispute Reporting'
        elif any(w in t for w in ['bank', 'checking', 'saving', 'account', 'deposit', 'withdrawal', 'branch', 'atm', 'wire transfer', 'overdraft']):
            return 'Bank Account services'
        else:
            return 'Others'

    def rule_based_priority(self, text):
        """Rule-based priority fallback"""
        t = text.lower()
        if any(w in t for w in ['urgent', 'fraud', 'stolen', 'unauthorized', 'emergency', 'immediately', 'critical', 'lawsuit']):
            return 'High'
        elif any(w in t for w in ['problem', 'issue', 'delay', 'dispute', 'wrong', 'error', 'fee', 'charge']):
            return 'Medium'
        else:
            return 'Low'

    def generate_ticket(self, text):
        """Complete structured ticket generation pipeline"""
        if not text or len(str(text).strip()) < 5:
            return {
                "title": "Invalid Ticket Request",
                "description": str(text),
                "category": "Others",
                "priority": "Low",
                "entities": {},
                "ml_used": False,
                "error": "Description is too short."
            }

        title = self.generate_title(text)
        category = self.predict_category(text)
        priority = self.predict_priority(text)
        entities = self.extract_entities(text)

        return {
            "title": title,
            "description": text,
            "category": category,
            "priority": priority,
            "entities": entities,
            "ml_used": self.models_loaded
        }

# Global singleton instance
ai_generator = AITicketGenerator()

def generate_ticket(text):
    """Entry point for ticket generation"""
    return ai_generator.generate_ticket(text)