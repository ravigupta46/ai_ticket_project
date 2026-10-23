"""create_agent.py
CLI utility to provision support agent accounts securely.
Usage:
    python create_agent.py <username>
    python create_agent.py
"""
import sys
import getpass
from database import init_db, create_user


def main():
    init_db()

    if len(sys.argv) > 1 and sys.argv[1].strip():
        username = sys.argv[1].strip()
    else:
        try:
            username = input("Enter agent username: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)

    if not username:
        print("Error: Username cannot be empty.")
        sys.exit(1)

    try:
        if sys.stdin.isatty():
            password = getpass.getpass(f"Enter password for agent '{username}' (min 8 chars): ")
        else:
            # Piped or redirected input (e.g. automated environments)
            password = sys.stdin.readline().rstrip("\r\n")
            if not password:
                password = getpass.getpass(f"Enter password for agent '{username}' (min 8 chars): ")
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        sys.exit(1)

    if len(password) < 8:
        print("Error: Password must be at least 8 characters long.")
        sys.exit(1)

    success, message = create_user(username, password, "agent")
    if success:
        print(f"[SUCCESS] Agent '{username}' created successfully.")
    else:
        print(f"[ERROR] Failed to create agent: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
