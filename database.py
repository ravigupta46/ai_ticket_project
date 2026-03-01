# database.py (FIXED)
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # USERS TABLE - Added UNIQUE constraint on username
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # TICKETS TABLE - Added created_at column
    c.execute("""
    CREATE TABLE IF NOT EXISTS tickets(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL,
        priority TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username)
    )
    """)

    # Check if we need to add created_at column to existing table
    try:
        c.execute("ALTER TABLE tickets ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        c.execute("ALTER TABLE tickets ADD COLUMN resolved_at TIMESTAMP")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

# Helper function to get database connection
def get_db():
    """Get database connection"""
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn
# Function to create a new ticket
# database.py - Update the create_ticket function

# database.py - More flexible create_ticket function

# database.py - Simplified working version

def create_ticket(username, title, description, category, priority):
    """Create a new ticket"""
    conn = get_db()
    c = conn.cursor()
    
    c.execute("""
    INSERT INTO tickets (username, title, description, category, priority, status, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, title, description, category, priority, 'Open', datetime.now()))
    
    ticket_id = c.lastrowid
    conn.commit()
    conn.close()
    return ticket_id


# Function to get user tickets
def get_user_tickets(username):
    conn = get_db()
    c = conn.cursor()
    
    c.execute("""
    SELECT * FROM tickets 
    WHERE username = ? 
    ORDER BY created_at DESC
    """, (username,))
    
    tickets = c.fetchall()
    conn.close()
    return tickets

# Function to get all tickets (for agents)
def get_all_tickets():
    conn = get_db()
    c = conn.cursor()
    
    c.execute("""
    SELECT * FROM tickets 
    ORDER BY 
        CASE priority 
            WHEN 'High' THEN 1 
            WHEN 'Medium' THEN 2 
            WHEN 'Low' THEN 3 
        END,
        created_at DESC
    """)
    
    tickets = c.fetchall()
    conn.close()
    return tickets

# Function to update ticket status
def update_ticket_status(ticket_id, status):
    conn = get_db()
    c = conn.cursor()
    
    resolved_at = datetime.now() if status == 'Resolved' else None
    
    c.execute("""
    UPDATE tickets 
    SET status = ?, resolved_at = COALESCE(?, resolved_at)
    WHERE id = ?
    """, (status, resolved_at, ticket_id))
    
    conn.commit()
    conn.close()
    return True

# Function to get ticket by ID
def get_ticket_by_id(ticket_id):
    conn = get_db()
    c = conn.cursor()
    
    c.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    ticket = c.fetchone()
    conn.close()
    return ticket

# Function to get ticket statistics
def get_ticket_stats():
    conn = get_db()
    c = conn.cursor()
    
    stats = {}
    
    # Total tickets
    c.execute("SELECT COUNT(*) FROM tickets")
    stats['total'] = c.fetchone()[0]
    
    # By status
    c.execute("SELECT status, COUNT(*) FROM tickets GROUP BY status")
    stats['by_status'] = dict(c.fetchall())
    
    # By priority
    c.execute("SELECT priority, COUNT(*) FROM tickets GROUP BY priority")
    stats['by_priority'] = dict(c.fetchall())
    
    # By category
    c.execute("SELECT category, COUNT(*) FROM tickets GROUP BY category")
    stats['by_category'] = dict(c.fetchall())
    
    conn.close()
    return stats