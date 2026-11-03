"""ai_engine/ticket_generator.py
Simple pure-ML ticket generator using scikit-learn pipelines and spaCy NER.
"""
import re
import pickle
import spacy
from model.config import CAT_MODEL_PATH, PRI_MODEL_PATH

COMMON_PRODUCTS = [
    'laptop', 'computer', 'pc', 'desktop', 'macbook', 'notebook',
    'mobile', 'phone', 'smartphone', 'iphone', 'android', 'tablet', 'ipad',
    'television', 'tv', 'monitor', 'display', 'screen',
    'headphones', 'earphones', 'airpods', 'speaker',
    'keyboard', 'mouse', 'charger', 'router', 'dongle', 'adapter'
]
PRODUCT_REGEX = re.compile(r'\b(?:' + '|'.join(COMMON_PRODUCTS) + r')\b', re.IGNORECASE)


class AITicketGenerator:
    """Singleton ML inference engine for category and priority prediction."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            with open(CAT_MODEL_PATH, "rb") as f:
                cls._instance.category_pipe = pickle.load(f)
            with open(PRI_MODEL_PATH, "rb") as f:
                cls._instance.priority_pipe = pickle.load(f)
            try:
                cls._instance.nlp = spacy.load("en_core_web_sm")
            except Exception:
                cls._instance.nlp = None
        return cls._instance

    def extract_entities(self, text: str) -> dict:
        """Run spaCy NER and keyword extraction on text."""
        entities = {'error_codes': [], 'organizations': [], 'amounts': [], 'dates': [], 'products': []}
        if not text:
            return entities

        # Product matching
        for m in PRODUCT_REGEX.finditer(text):
            p = m.group(0).capitalize()
            if p not in entities['products']:
                entities['products'].append(p)

        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG" and ent.text not in entities['organizations']:
                    entities['organizations'].append(ent.text)
                elif ent.label_ == "MONEY" and ent.text not in entities['amounts']:
                    entities['amounts'].append(ent.text)
                elif ent.label_ == "DATE" and ent.text not in entities['dates']:
                    entities['dates'].append(ent.text)

            # Code/ID extraction
            for token in doc:
                if any(char.isdigit() for char in token.text) and len(token.text) >= 4:
                    if token.text not in entities['error_codes']:
                        entities['error_codes'].append(token.text)

        return entities

    def generate_ticket_preview(self, text: str, subject: str = "") -> dict:
        """Pure-ML ticket classification and preview generation."""
        cleaned_text = text.strip()
        if len(cleaned_text) < 5:
            return {
                "title": "Invalid Ticket Request",
                "description": cleaned_text,
                "category": "Others",
                "priority": "Low",
                "entities": {"error_codes": [], "organizations": [], "amounts": [], "dates": [], "products": []}
            }

        full_text = (subject.strip() + " " + cleaned_text).strip() if subject else cleaned_text
        category = str(self.category_pipe.predict([full_text])[0])
        priority = str(self.priority_pipe.predict([full_text])[0])
        entities = self.extract_entities(full_text)

        words = cleaned_text.split()
        title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")

        return {
            "title": title,
            "description": cleaned_text,
            "category": category,
            "priority": priority,
            "entities": entities
        }


# Convenience functional APIs
def generate_ticket(text: str, subject: str = "") -> dict:
    return AITicketGenerator().generate_ticket_preview(text, subject)


def extract_entities(text: str) -> dict:
    return AITicketGenerator().extract_entities(text)
