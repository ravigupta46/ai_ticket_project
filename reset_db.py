# reset_db.py
import os
import sqlite3
from database import init_db

print("="*60)
print("⚠️  DATABASE RESET TOOL")
print("="*60)
print("\nThis will DELETE all existing data and create a fresh database!")
print("  - All users will be deleted")
print("  - All tickets will be deleted")
print("  - All tables will be recreated")

confirm = input("\nType 'YES' to continue: ")

if confirm == "YES":
    # Delete existing database
    db_file = "users.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"✅ Deleted old database: {db_file}")
    else:
        print("📂 No existing database found")
    
    # Create new database with updated schema
    print("\n🔄 Creating new database with updated schema...")
    init_db()
    
    print("\n" + "="*60)
    print("✅ DATABASE RESET COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📝 Next steps:")
    print("1. Run your Flask app: python app.py")
    print("2. Register new users")
    print("3. Start creating tickets")
else:
    print("\n❌ Operation cancelled")