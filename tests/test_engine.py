"""tests/test_engine.py
Simple, fresher-level unit tests for ResolvAI pipeline and engine.
"""
import pandas as pd
from model.preprocessing import clean_data, preprocess_and_split
from model.predict import predict_ticket
from ai_engine.ticket_generator import AITicketGenerator, generate_ticket


def test_clean_data_returns_dataframe():
    """Verify that clean_data returns a valid pandas DataFrame with required columns."""
    df = clean_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "text" in df.columns
    assert "category" in df.columns
    assert "priority" in df.columns


def test_preprocess_and_split_shapes():
    """Verify 80/20 train-test split shapes and sample alignment."""
    df = clean_data()
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = preprocess_and_split(df)
    assert len(X_train) == 3200
    assert len(X_test) == 800
    assert len(y_cat_train) == 3200
    assert len(y_pri_train) == 3200


def test_predict_ticket_returns_dict_with_keys():
    """Verify that predict_ticket produces expected keys and non-empty predictions."""
    sample_text = "Database connection error in production cluster. Customers cannot checkout."
    result = predict_ticket(text=sample_text, subject="Critical Server Outage")
    assert isinstance(result, dict)
    assert "input_text" in result
    assert "predicted_category" in result
    assert "predicted_priority" in result
    assert result["predicted_category"] != ""
    assert result["predicted_priority"] in ["High", "Medium", "Low"]


def test_ticket_generator_singleton_returns_same_instance():
    """Verify AITicketGenerator follows singleton pattern and returns the same instance."""
    instance1 = AITicketGenerator()
    instance2 = AITicketGenerator()
    assert instance1 is instance2

    ticket = generate_ticket("Cannot connect to wifi dongle and adapter is missing drivers")
    assert "category" in ticket
    assert "priority" in ticket
    assert "title" in ticket
    assert "entities" in ticket
