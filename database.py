
import sqlite3
from pathlib import Path
from datetime import datetime

# Database file location
PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "pcb_inspection.db"


# Connect to SQLite database
def get_connection():
    return sqlite3.connect(DB_PATH, timeout=10)


# Create inspection history table
def create_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                inspection_mode TEXT NOT NULL,
                image_path TEXT,
                defect_count INTEGER NOT NULL,
                defect_types TEXT,
                confidence_scores TEXT,
                status TEXT NOT NULL
            )
        """)


# Save an inspection record
def save_inspection(
    inspection_mode,
    image_path,
    defect_count,
    defect_types,
    confidence_scores,
    status
):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO inspections (
                timestamp,
                inspection_mode,
                image_path,
                defect_count,
                defect_types,
                confidence_scores,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            inspection_mode,
            str(image_path),
            int(defect_count),
            ", ".join(defect_types),
            ", ".join(f"{score:.1%}" for score in confidence_scores),
            status
        ))


# Retrieve inspection history
def get_inspections():
    with get_connection() as conn:
        return conn.execute("""
            SELECT *
            FROM inspections
            ORDER BY id DESC
        """).fetchall()


# Create the table when this file is imported or run
create_table()

print("SQLite database initialized successfully.")
print(f"Database location: {DB_PATH}")
