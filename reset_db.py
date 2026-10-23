"""reset_db.py
Database reset tool. Creates fresh schema and seeds test users with secure password hashes.
"""
import os
from database import init_db, create_user

DB_FILE = "users.db"


def reset_database(force: bool = False):
    """Delete old database and recreate fresh schema with seed accounts."""
    if not force:
        confirm = input("\nType 'YES' to delete all data and recreate the database: ")
        if confirm != "YES":
            print("Operation cancelled.")
            return

    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
            print(f"[RESET] Removed existing database: {DB_FILE}")
        except Exception as e:
            print(f"[WARN] Could not remove {DB_FILE}: {e}")

    init_db()
    # Seed development accounts with secure hashes
    create_user("demo_user", "password123", "user")
    create_user("demo_agent", "password123", "agent")
    print("[RESET] Fresh database created and seeded with test accounts ('demo_user', 'demo_agent').")


if __name__ == "__main__":
    reset_database()