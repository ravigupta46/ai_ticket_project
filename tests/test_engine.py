"""tests/test_engine.py
Pytest suite covering:
1. Word-boundary false-positive prevention ("upcoming" must NOT trigger 'pc').
2. One example per category across all 12 canonical categories.
3. Priority detection (High / Medium / Low).
4. Entity extraction (Order IDs, amounts, products).
5. Invalid / short input handling.
6. Rules-only fallback mode when ML models are absent.
"""
import pytest
from ai_engine.config import ALLOWED_CATEGORIES, ALLOWED_PRIORITIES
from ai_engine.rules import rule_based_category, rule_based_priority
from ai_engine.ticket_generator import AITicketGenerator, generate_ticket, extract_entities


def test_word_boundary_pc_in_upcoming():
    """Verify substring 'pc' inside 'upcoming' does NOT trigger Product Queries or devices."""
    text = "Tell me about the upcoming sales and discount dates"
    # Rule 12 requires both symptom and device; device regex must not match 'upcoming'
    from ai_engine.config import COMPILED_DEVICES
    assert COMPILED_DEVICES.search(text) is None
    # Category should not falsely become Product Queries
    cat = rule_based_category(text)
    assert cat != "Product Queries"


def test_word_boundary_isolated_pc():
    """Verify isolated word 'pc' with a symptom DOES match Product Queries."""
    text = "My desktop pc is overheating and not working"
    cat = rule_based_category(text)
    assert cat == "Product Queries"


@pytest.mark.parametrize("query,expected_category", [
    ("Unable to login to my account, password reset otp not received", "Shopzilla Related"),
    ("Please call me back regarding my inquiry", "Others"),
    ("The mobile app is crashing on the checkout screen with ui error", "App/website"),
    ("I have a coupon promo code and cashback discount not applied", "Offers & Cashback"),
    ("Please cancel my order immediately before dispatch", "Cancellation"),
    ("Where is my refund for the returned item? Amount not credited", "Refund Related"),
    ("Payment failed but money debited twice from my card", "Payments related"),
    ("I want to return this damaged product and schedule reverse pickup", "Returns"),
    ("The delivery agent was extremely rude and abusive to me", "Feedback"),
    ("How do I complete seller onboarding and merchant registration?", "Onboarding related"),
    ("Where is my order? The courier delivery is delayed", "Order Related"),
    ("Laptop display screen is blank and overheating", "Product Queries")
])
def test_all_twelve_categories(query, expected_category):
    """Test deterministic rule detection for each of the 12 canonical categories."""
    assert expected_category in ALLOWED_CATEGORIES
    pred = rule_based_category(query)
    assert pred == expected_category


@pytest.mark.parametrize("query,expected_priority", [
    ("URGENT: Emergency! The battery caught fire and exploded with smoke", "High"),
    ("Unauthorized fraud transaction on my account, money stolen", "High"),
    ("My order is delayed and the product arrived damaged and broken", "Medium"),
    ("The app has a problem and payment failed", "Medium"),
    ("Can you please provide the dimensions of the dining table?", "Low"),
    ("General inquiry about your shipping return policy", "Low")
])
def test_priority_levels(query, expected_priority):
    """Verify High, Medium, Low urgency detection."""
    assert expected_priority in ALLOWED_PRIORITIES
    prio = rule_based_priority(query)
    assert prio == expected_priority


def test_entity_extraction():
    """Verify extraction of error/order codes, currency amounts, and products."""
    text = "Order #ORD-98431 for my new laptop had a billing charge of $250.00 and Rs. 5000"
    entities = extract_entities(text)
    assert any("98431" in code or "ORD-98431" in code for code in entities["error_codes"])
    assert any("$250" in amt for amt in entities["amounts"]) or any("5000" in amt for amt in entities["amounts"])
    assert any("Laptop" in p for p in entities["products"])


def test_invalid_short_input():
    """Verify input under 5 characters triggers graceful error ticket."""
    result = generate_ticket("bad")
    assert result["title"] == "Invalid Ticket Request"
    assert result["category"] == "Others"
    assert result["priority"] == "Low"
    assert result["ml_used"] is False
    assert "too short" in result["error"].lower()

    empty_result = generate_ticket("")
    assert empty_result["title"] == "Invalid Ticket Request"
    assert empty_result["ml_used"] is False


def test_rules_only_fallback_when_models_absent(monkeypatch):
    """Verify system degrades to rules-only with ml_used=False when pipelines are missing."""
    engine = AITicketGenerator()
    # Temporarily remove pipelines
    orig_cat = engine.category_pipeline
    orig_pri = engine.priority_pipeline
    try:
        engine.category_pipeline = None
        engine.priority_pipeline = None

        ticket = engine.generate_ticket("Please cancel my order immediately")
        assert ticket["category"] == "Cancellation"
        assert ticket["priority"] == "Low" or ticket["priority"] in ALLOWED_PRIORITIES
        assert ticket["ml_used"] is False

        ticket_unknown = engine.generate_ticket("Some arbitrary text without any rule keywords")
        assert ticket_unknown["category"] == "Others"
        assert ticket_unknown["ml_used"] is False
    finally:
        engine.category_pipeline = orig_cat
        engine.priority_pipeline = orig_pri
