"""ai_engine/rules.py
High-precision, deterministic regex rules for customer support classification and priority.
Uses compiled word-boundary regexes from ai_engine.config.
"""
from ai_engine.config import (
    COMPILED_CATEGORY_RULES,
    COMPILED_SYMPTOMS,
    COMPILED_DEVICES,
    COMPILED_PRIORITY_HIGH,
    COMPILED_PRIORITY_MEDIUM,
)

def rule_based_category(text: str) -> str:
    """Evaluate ordered category rules with word-boundary regex matching.
    
    Returns one of the 12 canonical categories or 'Others' when no specific rule matches.
    Rule 12 (Product Queries) requires both a symptom/spec and a device keyword so it does
    not swallow unrelated queries before reaching the ML classifier.
    """
    if not text:
        return 'Others'
    t = str(text).strip()

    # Priority 1-11: Ordered domain rule table
    for category, regex in COMPILED_CATEGORY_RULES:
        if regex.search(t):
            return category

    # Rule 12: Product Queries (requires symptom AND device co-occurrence)
    if COMPILED_SYMPTOMS.search(t) and COMPILED_DEVICES.search(t):
        return 'Product Queries'

    return 'Others'

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
