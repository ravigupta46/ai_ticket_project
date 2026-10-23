"""tests/test_app.py
Flask integration and smoke test suite using test client:
1. Session expiry configuration check.
2. User registration and login with hashed password verification.
3. Ticket generation preview flow.
4. Confirm ticket validation: reject invalid category.
5. Successful ticket creation persists entities to database.
6. Ticket details route displays saved entities.
7. Agent status update with CSRF token.
"""
import re
import json
from datetime import timedelta
import pytest
from app import app
from database import init_db, get_db, authenticate_user, create_user


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = True
    init_db()
    with app.test_client() as client:
        yield client


def extract_csrf_token(html_text: str) -> str:
    """Helper to extract csrf_token value from rendered HTML."""
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html_text)
    if not match:
        match = re.search(r'value="([^"]+)"\s+name="csrf_token"', html_text)
    if not match:
        match = re.search(r"'X-CSRFToken':\s*'([^']+)'", html_text)
    return match.group(1) if match else ""


def test_session_configuration():
    """Verify session security and lifetime settings."""
    assert app.config['PERMANENT_SESSION_LIFETIME'] == timedelta(minutes=30)
    assert app.config['SESSION_COOKIE_HTTPONLY'] is True
    assert app.config['SESSION_COOKIE_SAMESITE'] == 'Lax'


def test_user_flow_and_validation(client):
    """Smoke test: register -> login -> preview -> reject invalid category -> confirm valid -> verify entities."""
    # 1. Register new user
    res = client.get("/register")
    assert res.status_code == 200
    csrf_token = extract_csrf_token(res.get_data(as_text=True))
    assert csrf_token != ""

    unique_user = "test_customer_99"
    raw_pwd = "SecurePassword!99"
    reg_res = client.post("/register", data={
        "csrf_token": csrf_token,
        "username": unique_user,
        "password": raw_pwd,
        "role": "user"
    }, follow_redirects=True)
    assert reg_res.status_code == 200

    # Verify password was stored as a hash, NOT plaintext
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (unique_user,))
    row = c.fetchone()
    conn.close()
    assert row is not None
    assert row["password"] != raw_pwd
    assert row["password"].startswith("scrypt:") or row["password"].startswith("pbkdf2:")

    # 2. Login
    login_page = client.get("/login")
    login_csrf = extract_csrf_token(login_page.get_data(as_text=True))
    login_res = client.post("/login", data={
        "csrf_token": login_csrf,
        "username": unique_user,
        "password": raw_pwd
    }, follow_redirects=True)
    assert login_res.status_code == 200
    assert "user_dashboard" in login_res.request.path

    # 3. Create Ticket preview
    create_page = client.get("/create_ticket")
    create_csrf = extract_csrf_token(create_page.get_data(as_text=True))
    description = "Order #ORD-77612 for my laptop was charged $350. Screen is blank and overheating."

    preview_res = client.post("/create_ticket", data={
        "csrf_token": create_csrf,
        "description": description
    })
    assert preview_res.status_code == 200
    preview_html = preview_res.get_data(as_text=True)
    assert "ResolvAI Classification Preview" in preview_html
    confirm_csrf = extract_csrf_token(preview_html)

    # 4. Reject invalid category on confirm
    reject_res = client.post("/confirm_ticket", data={
        "csrf_token": confirm_csrf,
        "title": "Hardware Defect",
        "description": description,
        "category": "InvalidNonExistentCategory",
        "priority": "High"
    }, follow_redirects=True)
    assert "create_ticket" in reject_res.request.path
    assert "Invalid category" in reject_res.get_data(as_text=True)

    # 5. Confirm with valid allowed category
    confirm_res = client.post("/confirm_ticket", data={
        "csrf_token": confirm_csrf,
        "title": "Overheating Laptop",
        "description": description,
        "category": "Product Queries",
        "priority": "High"
    }, follow_redirects=True)
    assert confirm_res.status_code == 200
    assert "user_dashboard" in confirm_res.request.path

    # 6. Verify ticket in database has entities column saved
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, entities FROM tickets WHERE username = ? ORDER BY id DESC LIMIT 1", (unique_user,))
    ticket_row = c.fetchone()
    conn.close()
    assert ticket_row is not None
    ticket_id = ticket_row["id"]
    entities = json.loads(ticket_row["entities"])
    assert any("77612" in code for code in entities.get("error_codes", []))
    assert any("Laptop" in p for p in entities.get("products", []))

    # 7. View ticket details and verify entities are rendered
    detail_res = client.get(f"/ticket/{ticket_id}")
    assert detail_res.status_code == 200
    detail_html = detail_res.get_data(as_text=True)
    assert "Extracted Information" in detail_html
    assert "Laptop" in detail_html


def test_agent_status_update_with_csrf(client):
    """Test agent updating ticket status with CSRF token header."""
    # Provision agent account via backend
    create_user("agent_smith_42", "agentPassword123!", "agent")

    login_page = client.get("/login")
    login_csrf = extract_csrf_token(login_page.get_data(as_text=True))
    client.post("/login", data={
        "csrf_token": login_csrf,
        "username": "agent_smith_42",
        "password": "agentPassword123!"
    })

    # Get dashboard to obtain CSRF token
    dash = client.get("/agent_dashboard")
    dash_html = dash.get_data(as_text=True)
    csrf_token = extract_csrf_token(dash_html)
    assert csrf_token != ""

    # Fetch a ticket ID from database
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM tickets LIMIT 1")
    row = c.fetchone()
    conn.close()

    if row:
        t_id = row["id"]
        # Post status update with X-CSRFToken header
        resp = client.post(
            f"/update_ticket_status/{t_id}",
            json={"status": "In Progress"},
            headers={"X-CSRFToken": csrf_token}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

        # Verify status in database
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT status FROM tickets WHERE id = ?", (t_id,))
        updated_status = c.fetchone()["status"]
        conn.close()
        assert updated_status == "In Progress"
