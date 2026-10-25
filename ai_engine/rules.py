"""ai_engine/rules.py
Urgency rules for support ticket priority detection.
"""
import re

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

def _build_regex(keywords):
    sorted_kw = sorted(keywords, key=len, reverse=True)
    pattern = r'\b(?:' + '|'.join(re.escape(k) for k in sorted_kw) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

COMPILED_PRIORITY_HIGH = _build_regex(PRIORITY_HIGH_KEYWORDS)
COMPILED_PRIORITY_MEDIUM = _build_regex(PRIORITY_MEDIUM_KEYWORDS)


def rule_based_priority(text: str) -> str:
    """Evaluate priority using compiled urgency regexes."""
    if not text:
        return 'Low'
    t = str(text).strip()
    if COMPILED_PRIORITY_HIGH.search(t):
        return 'High'
    if COMPILED_PRIORITY_MEDIUM.search(t):
        return 'Medium'
    return 'Low'
