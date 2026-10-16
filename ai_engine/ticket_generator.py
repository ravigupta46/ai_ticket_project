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

# Canonical customer support categories aligned with Customer_support_data.csv
CATEGORY_MAPPING = {
    'Product Queries': 'Product Queries',
    'Returns': 'Returns',
    'Order Related': 'Order Related',
    'Refund Related': 'Refund Related',
    'Payments related': 'Payments related',
    'Cancellation': 'Cancellation',
    'Shopzilla Related': 'Shopzilla Related',
    'Feedback': 'Feedback',
    'Offers & Cashback': 'Offers & Cashback',
    'App/website': 'App/website',
    'Onboarding related': 'Onboarding related',
    'Others': 'Others',
    # Legacy fallbacks
    'Bank Account services': 'Product Queries',
    'Credit card or prepaid card': 'Payments related',
    'Theft/Dispute Reporting': 'Returns',
    'Mortgage/Loan': 'Order Related',
    0: 'Product Queries',
    1: 'Returns',
    2: 'Order Related',
    3: 'Refund Related',
    4: 'Payments related'
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
                logger.info("[OK] Customer support category model & vectorizer loaded successfully")

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
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception as ex:
                    logger.warning(f"Could not load spaCy: {ex}")

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
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_entities(self, text):
        """Extract domain entities: products, order IDs, money amounts, dates, error codes"""
        entities = {
            'error_codes': [],
            'organizations': [],
            'amounts': [],
            'dates': [],
            'products': []
        }

        # 1. Product detection
        common_products = [
            'laptop', 'computer', 'pc', 'desktop', 'macbook', 'notebook',
            'mobile', 'phone', 'smartphone', 'iphone', 'android', 'tablet', 'ipad',
            'television', 'tv', 'led tv', 'oled tv', 'monitor', 'display', 'screen',
            'headphones', 'earphones', 'airpods', 'earbuds', 'speaker', 'soundbar',
            'watch', 'smartwatch', 'camera', 'dslr', 'lens',
            'keyboard', 'mouse', 'charger', 'adapter', 'cable', 'hard disk', 'ssd', 'ram',
            'refrigerator', 'fridge', 'washing machine', 'microwave', 'ac', 'air conditioner',
            'shoes', 'shirt', 'dress', 'jeans', 'tshirt', 'jacket', 't-shirt'
        ]
        text_lower = text.lower()
        for prod in common_products:
            if re.search(r'\b' + re.escape(prod) + r'\b', text_lower):
                entities['products'].append(prod.capitalize())

        # 2. Extract with spaCy if available
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
            except Exception:
                pass

        # 3. Regex patterns for Order IDs, reference codes, currency amounts
        order_patterns = [
            r'#\s?([A-Za-z0-9]{5,15})\b',
            r'\b(?:order|ord|ticket|txn|id)[-:\s#]+([A-Za-z0-9]{4,15})\b',
            r'\b[A-Z]{2,4}[-_]?\d{4,10}\b'
        ]
        for pattern in order_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                entities['error_codes'].append(m if isinstance(m, str) else m[0])

        currency_patterns = [
            r'(?:Rs\.?|INR|\$|€|£)\s?\d+(?:,\d{3})*(?:\.\d{2})?',
            r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:rupees|dollars|inr|usd)\b'
        ]
        for pattern in currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                entities['amounts'].append(m)

        # Deduplicate and limit to 5 per list
        for k in entities:
            entities[k] = list(dict.fromkeys(entities[k]))[:5]

        return entities

    def generate_title(self, text):
        """Generate a concise, professional title from complaint text"""
        if not text:
            return "Support Ticket"
        sentences = re.split(r'[.!?\n]', text)
        first_sentence = sentences[0].strip() if sentences and sentences[0].strip() else text[:60]
        
        if len(first_sentence) > 75:
            title = first_sentence[:75].rsplit(' ', 1)[0] + '...'
        else:
            title = first_sentence

        title = title.strip().capitalize()
        if not title.endswith(('.', '!', '?', '...')):
            title += '...'
        return title

    def rule_based_category(self, text):
        """
        High-precision rule-based category classifier based on customer support data.
        Maps requests to one of the 12 canonical customer support categories:
        - Product Queries
        - Returns
        - Order Related
        - Refund Related
        - Payments related
        - Cancellation
        - Shopzilla Related
        - Feedback
        - Offers & Cashback
        - App/website
        - Onboarding related
        - Others
        """
        t = text.lower()

        # 1. Shopzilla Account & Access (Login, signup, password, OTP, profile)
        if any(w in t for w in [
            'login', 'log in', 'signup', 'sign up', 'password reset', 'forgot password',
            'password', 'otp not received', 'update phone', 'update email', 'account deactivation',
            'shopzilla premium', 'membership', 'rewards points'
        ]):
            return 'Shopzilla Related'

        # 2. Others / Call requests
        if any(w in t for w in ['call me back', 'call back request', 'call disconnected', 'callback']):
            return 'Others'

        # 3. App/website (App crash, bug, website not opening)
        if any(w in t for w in [
            'app crash', 'app is crashing', 'crashes', 'website not loading',
            'app not opening', 'app freezing', 'bug in app', 'glitch on website',
            'ui error', 'error page', 'checkout crash'
        ]):
            return 'App/website'

        # 4. Offers & Cashback (Discounts, coupons, promo codes)
        if any(w in t for w in [
            'cashback', 'coupon', 'promo code', 'discount code', 'affiliate offer',
            'instant discount', 'voucher code', 'offer not applied'
        ]):
            return 'Offers & Cashback'

        # 5. Cancellation (Cancel order, stop shipment, not needed)
        if any(w in t for w in [
            'cancel order', 'cancellation', 'cancel my order', 'stop delivery',
            'not needed anymore', "don't want the product", 'ordered by mistake',
            'cancel shipment'
        ]):
            return 'Cancellation'

        # 6. Refund Related (Money refund, bank credit, COD refund)
        if any(w in t for w in [
            'refund', 'money back', 'amount not credited', 'cod refund',
            'refund status', 'refund pending', 'return payment', 'reimbursement',
            'where is my refund', 'refund not received'
        ]):
            return 'Refund Related'

        # 7. Payments related (Payment failed, debited, wallet, EMI, cards)
        if any(w in t for w in [
            'payment', 'money debited', 'money deducted', 'debited twice',
            'deducted twice', 'payment failed', 'transaction failed',
            'emi', 'paylater', 'pay later', 'wallet balance', 'net banking',
            'gateway error', 'billing error', 'double charged'
        ]):
            return 'Payments related'

        # 8. Returns (Defective/damaged product return, replacement, reverse pickup)
        if any(w in t for w in [
            'return', 'replace', 'replacement', 'exchange', 'reverse pickup',
            'wrong product', 'wrong item', 'damaged product', 'damaged item',
            'broken item', 'missing item', 'missing parts', 'technician visit',
            'defective product received', 'damaged parcel', 'exchange size'
        ]):
            return 'Returns'

        # 9. Feedback (Agent behaviour, rude delivery person, complaint on staff)
        if any(w in t for w in [
            'rude', 'abusive', 'behaviour', 'behavior', 'misbehaved',
            'unprofessional', 'delivery boy misbehaved', 'agent shouted',
            'pathetic customer support', 'worst executive'
        ]):
            return 'Feedback'

        # 10. Onboarding related (Seller registration, merchant)
        if any(w in t for w in [
            'seller onboarding', 'sell on shopzilla', 'merchant registration',
            'seller account', 'seller commission', 'gst verification'
        ]):
            return 'Onboarding related'

        # 11. Order Related (Order tracking, delayed delivery, invoice, installation)
        if any(w in t for w in [
            'delay', 'delayed', 'where is my order', 'order status', 'tracking',
            'track my package', 'courier', 'delivery boy', 'delivery date',
            'not delivered', 'dispatched', 'installation', 'demo', 'invoice copy',
            'bill copy', 'seller cancelled'
        ]):
            return 'Order Related'

        # 12. Product Queries (Hardware defect, overheating, display, specs, compatibility, warranty)
        hardware_symptoms = [
            'heating', 'overheating', 'heat up', 'heat', 'display not visible',
            'screen blank', 'screen is blank', 'display blank', 'screen not working',
            'not working', 'flickering', 'fan noise', 'motherboard', 'battery draining',
            'battery not charging', 'keyboard not working', 'camera not working',
            'touch not working', 'sound not coming', 'bluetooth not connecting',
            'wont turn on', "won't turn on", 'shut down automatically',
            'powers off', 'black screen', 'blue screen', 'ram upgrade', 'compatibility',
            'warranty period', 'claim warranty', 'user manual', 'product specification',
            'specifications', 'specs', 'warranty', 'guarantee'
        ]
        hardware_devices = [
            'laptop', 'computer', 'screen', 'display', 'mobile', 'phone', 'tv',
            'monitor', 'device', 'smartwatch', 'keyboard', 'camera', 'speaker',
            'headphone', 'charger', 'pc', 'tablet'
        ]
        
        if any(s in t for s in hardware_symptoms) or any(d in t for d in hardware_devices):
            return 'Product Queries'

        return 'Others'

    def predict_category(self, text):
        """
        Predict one of the 12 customer support categories:
        Combines domain intent disambiguation with trained TF-IDF classifier.
        """
        t = str(text).lower()
        if any(w in t for w in ['call me back', 'call back request', 'call disconnected', 'callback']):
            return 'Others'

        # 1. First check deterministic high-confidence rules
        rule_cat = self.rule_based_category(text)
        if rule_cat != 'Others':
            return rule_cat

        # 2. Use trained ML model if available
        if self.models_loaded and self.category_model is not None and self.vectorizer is not None:
            try:
                vec = self.vectorizer.transform([text])
                pred = self.category_model.predict(vec)[0]
                category_name = CATEGORY_MAPPING.get(pred, str(pred))
                return category_name
            except Exception as e:
                logger.error(f"Inference error in category prediction: {e}")
                return rule_cat

        return rule_cat

    def predict_priority(self, text):
        """Predict Priority (High, Medium, Low) based on customer urgency"""
        text_lower = text.lower()

        # High urgency: safety, fraud, severe heating, legal escalation
        critical_high = [
            'fire', 'smoke', 'shock', 'electric', 'blast', 'exploded',
            'burst', 'heating very badly', 'overheating badly',
            'urgent', 'emergency', 'asap', 'critical', 'immediately',
            'fraud', 'stolen', 'unauthorized', 'theft', 'skimming',
            'consumer court', 'police', 'legal notice', 'lawsuit',
            'abusive', 'harassment'
        ]
        if any(w in text_lower for w in critical_high):
            return 'High'

        # Medium urgency: product not working, heating, display, delay, refund pending
        critical_medium = [
            'not working', 'heating', 'display not visible', 'screen blank',
            'flickering', 'broken', 'damaged', 'defective', 'wrong product',
            'delayed', 'refund pending', 'payment failed', 'money debited',
            'missing item', 'app crash'
        ]
        if any(w in text_lower for w in critical_medium):
            return 'Medium'

        # Check trained priority model
        if self.priority_model is not None and self.priority_vectorizer is not None:
            try:
                vec = self.priority_vectorizer.transform([text])
                pred = self.priority_model.predict(vec)[0]
                return str(pred)
            except Exception as e:
                logger.error(f"Inference error in priority prediction: {e}")

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