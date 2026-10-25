"""ai_engine/ticket_generator.py
Triage engine: ML classification for category, priority rules/model, and entity extraction.
Maintains public API: generate_ticket(text) -> {title, description, category, priority, entities, ml_used}.
"""
import os
import re
import pickle
import logging
from typing import Dict, Any

from ai_engine.config import ALLOWED_CATEGORIES, ALLOWED_PRIORITIES
from ai_engine.rules import rule_based_priority

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

COMMON_PRODUCTS = [
    'laptop', 'computer', 'pc', 'desktop', 'macbook', 'notebook',
    'mobile', 'phone', 'smartphone', 'iphone', 'android', 'tablet', 'ipad',
    'television', 'tv', 'led tv', 'oled tv', 'monitor', 'display', 'screen',
    'headphones', 'earphones', 'airpods', 'earbuds', 'speaker', 'soundbar',
    'watch', 'smartwatch', 'camera', 'dslr', 'lens',
    'keyboard', 'mouse', 'charger', 'adapter', 'cable', 'hard disk', 'ssd', 'ram',
    'refrigerator', 'fridge', 'washing machine', 'microwave', 'ac', 'air conditioner',
    'shoes', 'shirt', 'dress', 'jeans', 'tshirt', 'jacket', 't-shirt'
]
PRODUCT_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(p) for p in sorted(COMMON_PRODUCTS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

ORDER_PATTERNS = [
    re.compile(r'#\s?([A-Za-z0-9]{5,15})\b'),
    re.compile(r'\b(?:order|ord|ticket|txn|id)[-:\s#]+([A-Za-z0-9]{4,15})\b', re.IGNORECASE),
    re.compile(r'\b[A-Z]{2,4}[-_]?\d{4,10}\b')
]

CURRENCY_PATTERNS = [
    re.compile(r'(?:Rs\.?|INR|\$|€|£)\s?\d+(?:,\d{3})*(?:\.\d{2})?', re.IGNORECASE),
    re.compile(r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:rupees|dollars|inr|usd)\b', re.IGNORECASE)
]


class AITicketGenerator:
    """Inference engine combining ML pipeline for categories, priority triage, and NER."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        self.category_pipeline = None
        self.priority_pipeline = None
        self.nlp = None
        self._load_pipelines()
        self._load_spacy()

    def _load_pipelines(self):
        cat_pipe_path = os.path.join(MODEL_DIR, 'category_pipeline.pkl')
        pri_pipe_path = os.path.join(MODEL_DIR, 'priority_pipeline.pkl')

        if os.path.exists(cat_pipe_path):
            try:
                with open(cat_pipe_path, 'rb') as f:
                    self.category_pipeline = pickle.load(f)
            except Exception as e:
                logger.error("Failed loading category_pipeline.pkl: %s", e)

        if os.path.exists(pri_pipe_path):
            try:
                with open(pri_pipe_path, 'rb') as f:
                    self.priority_pipeline = pickle.load(f)
            except Exception as e:
                logger.warning("Failed loading priority_pipeline.pkl: %s", e)

    def _load_spacy(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def extract_entities(self, text: str) -> Dict[str, list]:
        """Extract domain entities: order/reference IDs, amounts, dates, products, and orgs."""
        entities = {
            'error_codes': [],
            'organizations': [],
            'amounts': [],
            'dates': [],
            'products': []
        }
        if not text:
            return entities

        for match in PRODUCT_REGEX.finditer(text):
            entities['products'].append(match.group(0).capitalize())

        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == 'ORG' and ent.text.lower() not in [p.lower() for p in entities['products']]:
                        entities['organizations'].append(ent.text)
                    elif ent.label_ == 'DATE':
                        entities['dates'].append(ent.text)
                    elif ent.label_ == 'MONEY':
                        entities['amounts'].append(ent.text)
                    elif ent.label_ == 'PRODUCT':
                        entities['products'].append(ent.text)
            except Exception as e:
                logger.debug("spaCy extraction exception: %s", e)

        for pattern in ORDER_PATTERNS:
            for m in pattern.findall(text):
                entities['error_codes'].append(m if isinstance(m, str) else m[0])

        for pattern in CURRENCY_PATTERNS:
            for m in pattern.findall(text):
                entities['amounts'].append(m)

        for k in entities:
            entities[k] = list(dict.fromkeys(entities[k]))[:5]

        return entities

    def generate_title(self, text: str) -> str:
        """Create a concise, professional title from customer narrative."""
        if not text:
            return "Support Ticket"
        first = re.split(r'[.!?\n]', text)[0].strip() or text[:60]
        title = (first[:75].rsplit(' ', 1)[0] + '...') if len(first) > 75 else first
        title = title.strip().capitalize()
        return title if title.endswith(('.', '!', '?', '...')) else title + '...'

    def predict_category(self, text: str) -> str:
        """ML category prediction using TF-IDF + Logistic Regression pipeline."""
        if self.category_pipeline is not None:
            try:
                pred = self.category_pipeline.predict([text])[0]
                if pred in ALLOWED_CATEGORIES:
                    return pred
            except Exception as e:
                logger.error("Category ML inference error: %s", e)
        return 'Others'

    def predict_priority(self, text: str) -> str:
        """Priority triage: rules first, ML fallback."""
        rule_prio = rule_based_priority(text)
        if rule_prio != 'Low':
            return rule_prio

        if self.priority_pipeline is not None:
            try:
                pred = self.priority_pipeline.predict([text])[0]
                if pred in ALLOWED_PRIORITIES:
                    return pred
            except Exception as e:
                logger.error("Priority ML inference error: %s", e)

        return rule_prio

    def generate_ticket(self, text: str) -> Dict[str, Any]:
        """Public API generating structured ticket from free-form text."""
        if not text or len(str(text).strip()) < 5:
            return {
                "title": "Invalid Ticket Request",
                "description": str(text or ""),
                "category": "Others",
                "priority": "Low",
                "entities": {},
                "ml_used": False,
                "error": "Description is too short."
            }

        text_str = str(text).strip()
        category = self.predict_category(text_str)
        priority = self.predict_priority(text_str)
        entities = self.extract_entities(text_str)
        ml_used = bool(self.category_pipeline is not None)

        return {
            "title": self.generate_title(text_str),
            "description": text_str,
            "category": category,
            "priority": priority,
            "entities": entities,
            "ml_used": ml_used
        }


# Module singleton instance
ai_generator = AITicketGenerator()

def generate_ticket(text: str) -> Dict[str, Any]:
    """Public functional entry point for ticket generation."""
    return ai_generator.generate_ticket(text)

def extract_entities(text: str) -> Dict[str, list]:
    """Server-side entity extraction helper."""
    return ai_generator.extract_entities(text)
