"""
Database connection module for ML Cost Governance Framework.
Provides three levels of database access:
1. SQLAlchemy engine - for pandas and simple queries
2. Session factory - for transactions and ORM-style work
3. Raw psycopg2 - for when you need direct control
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import psycopg2
import sys
import os

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_URL, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# ── 1. SQLAlchemy engine ─────────────────────────────────────────────
# Used by: pandas read_sql/to_sql, simple queries
# echo=False prevents SQL statements from flooding your console
engine = create_engine(DB_URL, echo=False)

# ── 2. Session factory ───────────────────────────────────────────────
# Used by: transactions, inserts/updates that need rollback capability
SessionLocal = sessionmaker(bind=engine)

# ── 3. Raw psycopg2 connection ───────────────────────────────────────
# Used by: complex operations, COPY commands, when you need a raw connection
def get_raw_connection():
    """Return a raw psycopg2 connection for low-level operations"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# ── Helper function to get a session ─────────────────────────────────
def get_db_session():
    """Get a new database session (use in 'with' blocks)"""
    return SessionLocal()

# ── Test connection ──────────────────────────────────────────────────
def test_connection():
    """Verify all connection methods work"""
    results = {}
    
    # Test 1: Engine connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            value = result.scalar()
            results['engine'] = f"✅ Working (returned {value})"
    except Exception as e:
        results['engine'] = f"❌ Failed: {e}"
    
    # Test 2: Session
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1")).scalar()
        db.close()
        results['session'] = f"✅ Working (returned {result})"
    except Exception as e:
        results['session'] = f"❌ Failed: {e}"
    
    # Test 3: Raw connection
    try:
        conn = get_raw_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        results['raw'] = f"✅ Working (returned {result})"
    except Exception as e:
        results['raw'] = f"❌ Failed: {e}"
    
    # Print results
    print("\n" + "="*50)
    print("DATABASE CONNECTION TEST RESULTS")
    print("="*50)
    for method, result in results.items():
        print(f"{method.upper():10} : {result}")
    print("="*50)
    
    # Return True if all passed
    return all("✅" in r for r in results.values())

# ── Quick check when script runs directly ────────────────────────────
if __name__ == "__main__":
    test_connection()