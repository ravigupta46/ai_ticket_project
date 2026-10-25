"""tests/test_engine.py
Test suite covering:
1. ML category prediction.
2. Priority detection (High / Medium / Low).
3. Entity extraction (Order IDs, amounts, products).
4. Invalid / short input handling.
5. Fallback mode when model pipeline is absent.
"""
import pytest
from ai_engine.config import ALLOWED_CATEGORIES, ALLOWED_PRIORITIES
from ai_engine.rules import rule_based_priority
from ai_engine.ticket_generator import AITicketGenerator, generate_ticket, extract_entities


def test_category_prediction():
    """Verify ML category prediction returns a valid allowed category."""
    ticket = generate_ticket("I need to return this damaged item and get an exchange")
    assert ticket["category"] in ALLOWED_CATEGORIES
    assert ticket["category"] == "Returns"


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


def test_pipeline_absent_fallback():
    """Verify system defaults gracefully to 'Others' with ml_used=False when pipeline is absent."""
    engine = AITicketGenerator()
    orig_cat = engine.category_pipeline
    try:
        engine.category_pipeline = None
        ticket = engine.generate_ticket("Please cancel my order immediately")
        assert ticket["category"] == "Others"
        assert ticket["ml_used"] is False
    finally:
        engine.category_pipeline = orig_cat
