"""model/predict.py
Load saved pipelines and predict category and priority for a single ticket.
"""
import pickle
from model.config import CAT_MODEL_PATH, PRI_MODEL_PATH


def load_models():
    """Load CAT_MODEL_PATH and PRI_MODEL_PATH with pickle.
    Return (category_pipe, priority_pipe)."""
    with open(CAT_MODEL_PATH, "rb") as f:
        category_pipe = pickle.load(f)
    with open(PRI_MODEL_PATH, "rb") as f:
        priority_pipe = pickle.load(f)
    return category_pipe, priority_pipe


def predict_ticket(text, subject="", category_pipe=None, priority_pipe=None) -> dict:
    """Combine subject + text, predict category and priority.
    Return {'input_text', 'predicted_category', 'predicted_priority'}."""
    if category_pipe is None or priority_pipe is None:
        cat_p, pri_p = load_models()
        category_pipe = category_pipe or cat_p
        priority_pipe = priority_pipe or pri_p

    full_text = (subject.strip() + " " + text.strip()).strip() if subject else text.strip()
    if not full_text:
        return {
            "input_text": "",
            "predicted_category": "Others",
            "predicted_priority": "Low"
        }

    cat_pred = str(category_pipe.predict([full_text])[0])
    pri_pred = str(priority_pipe.predict([full_text])[0])

    return {
        "input_text": full_text[:120] + ("..." if len(full_text) > 120 else ""),
        "predicted_category": cat_pred,
        "predicted_priority": pri_pred
    }


if __name__ == "__main__":
    cat_pipe, pri_pipe = load_models()
    sample_tickets = [
        ("Production database outage: SQL server is unreachable for all customers.", "Critical server incident"),
        ("I was double billed on my visa invoice for subscription renew.", "Billing Question"),
        ("How do I return my damaged parcel and get a replacement item?", "Broken item return")
    ]
    for body, subj in sample_tickets:
        res = predict_ticket(body, subj, cat_pipe, pri_pipe)
        print("-" * 60)
        print(f"Subject:            {subj}")
        print(f"Predicted Category: {res['predicted_category']}")
        print(f"Predicted Priority: {res['predicted_priority']}")
