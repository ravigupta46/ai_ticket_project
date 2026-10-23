"""reset_db.py
Database reset tool. Creates fresh schema.
Demo accounts are only seeded if ALLOW_DEMO_USERS=1 is set in the environment.
"""
import os
from database import init_db, create_user

DB_FILE = "users.db"


def reset_database(force: bool = False):
    """Delete old database and recreate fresh schema."""
    if not force:
        confirm = input("\nType 'YES' to delete all data and recreate the database: ")
        if confirm != "YES":
            print("Operation cancelled.")
            return

    target = os.environ.get("DATABASE_PATH", DB_FILE)
    if os.path.exists(target):
        try:
            os.remove(target)
            print(f"[RESET] Removed existing database: {target}")
        except Exception as e:
            print(f"[WARN] Could not remove {target}: {e}")

    init_db()

    if os.environ.get("ALLOW_DEMO_USERS") == "1":
        create_user("demo_user", "password123", "user")
        create_user("demo_agent", "password123", "agent")
        print("[RESET] Fresh database created and seeded with demo accounts (ALLOW_DEMO_USERS=1).")
    else:
        print("[RESET] Fresh database created with empty user table.")
        print("        Use `python create_agent.py <username>` to provision an agent account.")


if __name__ == "__main__":
    reset_database()