"""database.py
SQLite database interface for ResolvAI tickets and users.
Features safe migrations, parameterized queries, and transparent password hash upgrades.
"""
import os
import sqlite3
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = "users.db"


def get_db(db_name=None):
    """Return database connection with Row factory."""
    target = db_name or os.environ.get("DATABASE_PATH", DB_NAME)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_name=None):
    """Initialize tables and run non-destructive schema migrations."""
    conn = get_db(db_name)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'agent')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL,
        priority TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Open',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP,
        entities TEXT DEFAULT '{}',
        FOREIGN KEY (username) REFERENCES users(username)
    )
    """)

    # Non-destructive migrations: inspect existing columns before adding
    c.execute("PRAGMA table_info(tickets)")
    cols = {row["name"] for row in c.fetchall()}

    if "created_at" not in cols:
        c.execute("ALTER TABLE tickets ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if "resolved_at" not in cols:
        c.execute("ALTER TABLE tickets ADD COLUMN resolved_at TIMESTAMP")
    if "entities" not in cols:
        c.execute("ALTER TABLE tickets ADD COLUMN entities TEXT DEFAULT '{}'")

    # Upgrade any legacy plaintext passwords to secure werkzeug hashes
    try:
        c.execute("SELECT id, password FROM users")
        for row in c.fetchall():
            pwd = row["password"]
            if pwd and not any(pwd.startswith(prefix) for prefix in ("scrypt:", "pbkdf2:", "argon2:")):
                c.execute("UPDATE users SET password = ? WHERE id = ?", (generate_password_hash(pwd), row["id"]))
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


def create_user(username, password, role):
    """Register a new user with hashed password."""
    if role not in ("user", "agent"):
        return False, "Invalid role"

    conn = get_db()
    c = conn.cursor()
    hashed = generate_password_hash(password)
    try:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username.strip(), hashed, role)
        )
        conn.commit()
        return True, "Registration successful! Please login."
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    finally:
        conn.close()


def authenticate_user(username, password):
    """Authenticate user with automatic transparent upgrade for legacy plaintext rows."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username = ?", (username.strip(),))
    user = c.fetchone()

    if not user:
        conn.close()
        return None

    stored = user["password"]
    role = user["role"]

    if check_password_hash(stored, password):
        conn.close()
        return {"username": username, "role": role}

    # Transparent legacy migration check
    if stored == password:
        new_hash = generate_password_hash(password)
        c.execute("UPDATE users SET password = ? WHERE username = ?", (new_hash, username.strip()))
        conn.commit()
        conn.close()
        return {"username": username, "role": role}

    conn.close()
    return None


def create_ticket(username, title, description, category, priority, entities=None):
    """Create a new ticket with serialized entities JSON."""
    conn = get_db()
    c = conn.cursor()
    entities_str = json.dumps(entities) if isinstance(entities, dict) else (entities or '{}')

    c.execute("""
    INSERT INTO tickets (username, title, description, category, priority, status, created_at, entities)
    VALUES (?, ?, ?, ?, ?, 'Open', ?, ?)
    """, (username, title, description, category, priority, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), entities_str))

    ticket_id = c.lastrowid
    conn.commit()
    conn.close()
    return ticket_id


def get_user_tickets(username):
    """Get tickets submitted by a specific user."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM tickets WHERE username = ? ORDER BY created_at DESC", (username,))
    tickets = c.fetchall()
    conn.close()
    return tickets


def get_all_tickets():
    """Get all tickets sorted by priority urgency and creation date."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
    SELECT * FROM tickets 
    ORDER BY 
        CASE priority 
            WHEN 'High' THEN 1 
            WHEN 'Medium' THEN 2 
            WHEN 'Low' THEN 3 
            ELSE 4
        END,
        created_at DESC
    """)
    tickets = c.fetchall()
    conn.close()
    return tickets


def update_ticket_status(ticket_id, status):
    """Update ticket resolution status."""
    conn = get_db()
    c = conn.cursor()
    resolved_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S') if status == 'Resolved' else None

    c.execute("""
    UPDATE tickets 
    SET status = ?, resolved_at = COALESCE(?, resolved_at)
    WHERE id = ?
    """, (status, resolved_at, ticket_id))

    success = c.rowcount > 0
    conn.commit()
    conn.close()
    return success


def get_ticket_by_id(ticket_id):
    """Fetch single ticket by ID."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    ticket = c.fetchone()
    conn.close()
    return ticket


def get_ticket_stats():
    """Aggregate statistics for agent operations console."""
    conn = get_db()
    c = conn.cursor()
    stats = {}

    c.execute("SELECT COUNT(*) FROM tickets")
    stats['total'] = c.fetchone()[0]

    for key, col in [('by_status', 'status'), ('by_priority', 'priority'), ('by_category', 'category')]:
        c.execute(f"SELECT {col}, COUNT(*) FROM tickets GROUP BY {col}")
        stats[key] = dict(c.fetchall())

    conn.close()
    return stats