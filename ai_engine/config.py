"""ai_engine/config.py
Shared configuration, canonical taxonomy, rule definitions, and keyword tables.
Single source of truth for categories, priorities, and rule regexes across training,
inference, and validation.
"""
import re

ALLOWED_CATEGORIES = [
    'Product Queries',
    'Returns',
    'Order Related',
    'Refund Related',
    'Payments related',
    'Cancellation',
    'Shopzilla Related',
    'Feedback',
    'Offers & Cashback',
    'App/website',
    'Onboarding related',
    'Others'
]

ALLOWED_PRIORITIES = ['High', 'Medium', 'Low']

# 11 direct category keyword rules (ordered by precedence)
CATEGORY_RULES_DEF = [
    ('Shopzilla Related', [
        'login', 'log in', 'signup', 'sign up', 'password reset', 'forgot password',
        'password', 'otp not received', 'update phone', 'update email', 'account deactivation',
        'shopzilla premium', 'membership', 'rewards points'
    ]),
    ('Others', [
        'call me back', 'call back request', 'call disconnected', 'callback'
    ]),
    ('App/website', [
        'app crash', 'app is crashing', 'crashes', 'website not loading',
        'app not opening', 'app freezing', 'bug in app', 'glitch on website',
        'ui error', 'error page', 'checkout crash'
    ]),
    ('Offers & Cashback', [
        'cashback', 'coupon', 'promo code', 'discount code', 'affiliate offer',
        'instant discount', 'voucher code', 'offer not applied'
    ]),
    ('Cancellation', [
        'cancel order', 'cancellation', 'cancel my order', 'stop delivery',
        'not needed anymore', "don't want the product", 'ordered by mistake',
        'cancel shipment'
    ]),
    ('Refund Related', [
        'refund', 'money back', 'amount not credited', 'cod refund',
        'refund status', 'refund pending', 'return payment', 'reimbursement',
        'where is my refund', 'refund not received'
    ]),
    ('Payments related', [
        'payment', 'money debited', 'money deducted', 'debited twice',
        'deducted twice', 'payment failed', 'transaction failed',
        'emi', 'paylater', 'pay later', 'wallet balance', 'net banking',
        'gateway error', 'billing error', 'double charged'
    ]),
    ('Returns', [
        'return', 'replace', 'replacement', 'exchange', 'reverse pickup',
        'wrong product', 'wrong item', 'damaged product', 'damaged item',
        'broken item', 'missing item', 'missing parts', 'technician visit',
        'defective product received', 'damaged parcel', 'exchange size'
    ]),
    ('Feedback', [
        'rude', 'abusive', 'behaviour', 'behavior', 'misbehaved',
        'unprofessional', 'delivery boy misbehaved', 'agent shouted',
        'pathetic customer support', 'worst executive'
    ]),
    ('Onboarding related', [
        'seller onboarding', 'sell on shopzilla', 'merchant registration',
        'seller account', 'seller commission', 'gst verification'
    ]),
    ('Order Related', [
        'delay', 'delayed', 'where is my order', 'order status', 'tracking',
        'track my package', 'courier', 'delivery boy', 'delivery date',
        'not delivered', 'dispatched', 'installation', 'demo', 'invoice copy',
        'bill copy', 'seller cancelled'
    ]),
]

HARDWARE_SYMPTOMS = [
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

HARDWARE_DEVICES = [
    'laptop', 'computer', 'screen', 'display', 'mobile', 'phone', 'tv',
    'monitor', 'device', 'smartwatch', 'keyboard', 'camera', 'speaker',
    'headphone', 'charger', 'pc', 'tablet'
]

# Shared Priority Keywords for training label generation AND runtime inference
PRIORITY_HIGH_KEYWORDS = [
    'fire', 'smoke', 'shock', 'electric', 'electric shock', 'blast', 'exploded',
    'burst', 'heating very badly', 'overheating badly', 'urgent', 'emergency',
    'asap', 'critical', 'immediately', 'fraud', 'stolen', 'unauthorized',
    'theft', 'skimming', 'consumer court', 'police', 'legal notice', 'lawsuit',
    'penalty', 'abusive', 'harassment', 'double debited', 'money stolen'
]

PRIORITY_MEDIUM_KEYWORDS = [
    'not working', 'heating', 'display not visible', 'screen blank',
    'flickering', 'broken', 'damaged', 'defective', 'wrong product',
    'delayed', 'delay', 'refund pending', 'payment failed', 'money debited',
    'missing item', 'missing', 'app crash', 'not delivered', 'problem',
    'issue', 'difficulty', 'trouble', 'help', 'assist', 'slow', 'dispute', 'error'
]

def build_regex(keywords):
    """Compile keywords into a word-boundary regex sorted by length descending."""
    sorted_kw = sorted(keywords, key=len, reverse=True)
    pattern = r'\b(?:' + '|'.join(re.escape(k) for k in sorted_kw) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

# Precompile regex tables once at import
COMPILED_CATEGORY_RULES = [(cat, build_regex(kw)) for cat, kw in CATEGORY_RULES_DEF]
COMPILED_SYMPTOMS = build_regex(HARDWARE_SYMPTOMS)
COMPILED_DEVICES = build_regex(HARDWARE_DEVICES)
COMPILED_PRIORITY_HIGH = build_regex(PRIORITY_HIGH_KEYWORDS)
COMPILED_PRIORITY_MEDIUM = build_regex(PRIORITY_MEDIUM_KEYWORDS)
