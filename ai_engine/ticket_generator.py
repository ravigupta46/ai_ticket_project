# ai_engine/ticket_generator.py (UPDATED with correct path)
import pickle
import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model')  # Changed from 'models' to 'model'

class AITicketGenerator:
    def __init__(self):
        self.models_loaded = False
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            # Check if model directory exists
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model directory not found: {MODEL_PATH}")
                return
            
            # Load category model
            category_model_file = os.path.join(MODEL_PATH, 'category_model.pkl')
            if os.path.exists(category_model_file):
                with open(category_model_file, 'rb') as f:
                    self.category_model = pickle.load(f)
                logger.info("✅ Category model loaded")
            else:
                logger.warning(f"Category model not found at {category_model_file}")
                return
            
            # Load priority model
            priority_model_file = os.path.join(MODEL_PATH, 'priority_model.pkl')
            if os.path.exists(priority_model_file):
                with open(priority_model_file, 'rb') as f:
                    self.priority_model = pickle.load(f)
                logger.info("✅ Priority model loaded")
            else:
                logger.warning(f"Priority model not found at {priority_model_file}")
                return
            
            # Load vectorizer
            vectorizer_file = os.path.join(MODEL_PATH, 'vectorizer.pkl')
            if os.path.exists(vectorizer_file):
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("✅ Vectorizer loaded")
            else:
                logger.warning(f"Vectorizer not found at {vectorizer_file}")
                return
            
            # Load spaCy for NER
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded")
            except:
                logger.warning("spaCy model not found, downloading...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
            
            self.models_loaded = True
            logger.info("✅ All models loaded successfully from 'model' folder")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("Falling back to rule-based system")
            self.models_loaded = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-z0-9\s\.\-_$%]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_entities(self, text):
        """Extract entities using NER and patterns"""
        entities = {
            'error_codes': [],
            'organizations': [],
            'amounts': [],
            'dates': [],
            'products': []
        }
        
        # If spaCy is loaded, use it
        if hasattr(self, 'nlp'):
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
        
        # Pattern-based extraction for error codes
        error_patterns = [
            r'\b[A-Z]+[-_]?\d+\b',           # ERROR-123
            r'\b0x[a-fA-F0-9]+\b',            # 0x7B
            r'\b\d{3}[-_]?\d{3}[-_]?\d{4}\b',  # Phone-like numbers
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, text)
            entities['error_codes'].extend(matches)
        
        # Remove duplicates and limit
        for key in entities:
            entities[key] = list(set(entities[key]))[:5]
        
        return entities
    
    def generate_title(self, text):
        """Generate a concise title"""
        # Take first sentence or first 60 chars
        sentences = re.split('[.!?]', text)
        if sentences and sentences[0].strip():
            title = sentences[0].strip()
        else:
            title = text[:60]
        
        # Clean and capitalize
        title = title[:70].capitalize()
        if not title.endswith(('.', '!', '?')):
            title += '...'
        
        return title
    
    def predict_category(self, text):
        """Predict category using ML model"""
        if not self.models_loaded:
            return self.rule_based_category(text)
        
        try:
            text_clean = self.preprocess_text(text)
            text_vec = self.vectorizer.transform([text_clean])
            category = self.category_model.predict(text_vec)[0]
            logger.info(f"ML Predicted category: {category}")
            return category
        except Exception as e:
            logger.error(f"Error in category prediction: {e}")
            return self.rule_based_category(text)
    
    def predict_priority(self, text):
        """Predict priority using ML model"""
        if not self.models_loaded:
            return self.rule_based_priority(text)
        
        try:
            text_clean = self.preprocess_text(text)
            text_vec = self.vectorizer.transform([text_clean])
            priority = self.priority_model.predict(text_vec)[0]
            
            # Override with rule-based for critical cases
            urgent_words = ['urgent', 'emergency', 'critical', 'asap', 'immediately']
            if any(word in text.lower() for word in urgent_words):
                return 'High'
            
            logger.info(f"ML Predicted priority: {priority}")
            return priority
        except Exception as e:
            logger.error(f"Error in priority prediction: {e}")
            return self.rule_based_priority(text)
    
    def rule_based_category(self, text):
        """Fallback rule-based category assignment"""
        text = text.lower()
        
        if any(word in text for word in ['laptop', 'computer', 'keyboard', 'mouse', 'screen', 'hardware']):
            return 'Hardware'
        elif any(word in text for word in ['network', 'wifi', 'internet', 'connect', 'vpn']):
            return 'Network'
        elif any(word in text for word in ['password', 'login', 'account', 'access', 'permission']):
            return 'Access'
        elif any(word in text for word in ['software', 'app', 'program', 'install', 'update']):
            return 'Software'
        elif any(word in text for word in ['email', 'outlook', 'thunderbird', 'mail']):
            return 'Email'
        else:
            return 'Other'
    
    def rule_based_priority(self, text):
        """Fallback rule-based priority assignment"""
        text = text.lower()
        
        if any(word in text for word in ['error', 'not working', 'broken', 'down', 'urgent']):
            return 'High'
        elif any(word in text for word in ['problem', 'issue', 'help', 'slow']):
            return 'Medium'
        else:
            return 'Low'
    
    def generate_ticket(self, text):
        """Main method to generate ticket"""
        if not text or len(text.strip()) < 10:
            return {
                "title": "Invalid Input",
                "description": text,
                "category": "Other",
                "priority": "Low",
                "entities": {},
                "error": "Description too short"
            }
        
        # Get predictions
        category = self.predict_category(text)
        priority = self.predict_priority(text)
        entities = self.extract_entities(text)
        title = self.generate_title(text)
        
        # Create ticket
        ticket = {
            "title": title,
            "description": text,
            "category": category,
            "priority": priority,
            "entities": entities,
            "ml_used": self.models_loaded
        }
        
        return ticket

# Initialize global instance
ai_generator = AITicketGenerator()

def generate_ticket(text):
    """Wrapper function"""
    return ai_generator.generate_ticket(text)