"""tests/test_register.py
Pytest suite verifying privilege escalation prevention and role boundaries:
1. POST /register with role=agent must store role='user'.
2. POST /register with short password (< 8 chars) is rejected and creates no row.
3. create_user(..., 'admin') returns False.
4. Logged-in user redirects to /user_dashboard; GET /agent_dashboard as user redirects to /login.
5. Agent created via create_user(..., 'agent') reaches /agent_dashboard.
Uses an isolated temporary SQLite database for zero production DB mutation.
"""
import os
import re
import pytest
from app import app
from database import init_db, get_db, create_user


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provide an isolated temporary database for each test."""
    db_file = str(tmp_path / "test_privilege.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)
    init_db(db_file)
    yield db_file
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass


@pytest.fixture
def client(temp_db):
    """Flask test client configured with temporary DB and CSRF enabled."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = True
    with app.test_client() as test_client:
        yield test_client


def extract_csrf_token(html_text: str) -> str:
    """Helper to extract csrf_token from rendered template."""
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html_text)
    if not match:
        match = re.search(r'value="([^"]+)"\s+name="csrf_token"', html_text)
    return match.group(1) if match else ""


def test_public_registration_cannot_escalate_to_agent(client):
    """POST /register with role=agent must be ignored; stored role must be 'user'."""
    reg_page = client.get("/register")
    csrf_token = extract_csrf_token(reg_page.get_data(as_text=True))
    assert csrf_token != ""

    response = client.post("/register", data={
        "csrf_token": csrf_token,
        "username": "evil_hacker",
        "password": "password123",
        "role": "agent"
    }, follow_redirects=True)

    assert response.status_code == 200

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username, role FROM users WHERE username = ?", ("evil_hacker",))
    user = c.fetchone()
    conn.close()

    assert user is not None
    assert user["role"] == "user"
    assert user["role"] != "agent"


def test_register_short_password_rejected(client):
    """POST /register with password < 8 chars must be rejected and create no account."""
    reg_page = client.get("/register")
    csrf_token = extract_csrf_token(reg_page.get_data(as_text=True))

    response = client.post("/register", data={
        "csrf_token": csrf_token,
        "username": "short_user",
        "password": "short"
    }, follow_redirects=True)

    assert response.status_code == 200
    assert "at least 8 characters" in response.get_data(as_text=True)

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", ("short_user",))
    row = c.fetchone()
    conn.close()

    assert row is None


def test_create_user_invalid_role():
    """create_user with an unauthorized role (e.g. 'admin') must return False."""
    success, msg = create_user("superadmin", "validPassword123", "admin")
    assert success is False
    assert "Invalid role" in msg


def test_user_cannot_access_agent_dashboard(client):
    """Login as customer redirects to /user_dashboard; GET /agent_dashboard redirects to /login."""
    create_user("legit_user", "password123", "user")

    login_page = client.get("/login")
    csrf_token = extract_csrf_token(login_page.get_data(as_text=True))

    login_res = client.post("/login", data={
        "csrf_token": csrf_token,
        "username": "legit_user",
        "password": "password123"
    }, follow_redirects=False)

    assert login_res.status_code == 302
    assert "/user_dashboard" in login_res.headers["Location"]

    # Now attempt to access agent dashboard as this user
    agent_dash_res = client.get("/agent_dashboard", follow_redirects=False)
    assert agent_dash_res.status_code == 302
    assert "/login" in agent_dash_res.headers["Location"]


def test_agent_can_access_agent_dashboard(client):
    """Agent created through create_user(..., 'agent') reaches /agent_dashboard."""
    create_user("trusted_agent", "password123", "agent")

    login_page = client.get("/login")
    csrf_token = extract_csrf_token(login_page.get_data(as_text=True))

    login_res = client.post("/login", data={
        "csrf_token": csrf_token,
        "username": "trusted_agent",
        "password": "password123"
    }, follow_redirects=False)

    assert login_res.status_code == 302
    assert "/agent_dashboard" in login_res.headers["Location"]

    dash_page = client.get("/agent_dashboard")
    assert dash_page.status_code == 200
    assert "Support Operations Console" in dash_page.get_data(as_text=True)
