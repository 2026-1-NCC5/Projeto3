import sqlite3
import os
from pathlib import Path
from datetime import datetime

# Define the database path
DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "food_inventory.db"

def init_db():
    """Initializes the database and creates the necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the food_items table with track_id, confidence and user_id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            track_id INTEGER,
            confidence REAL,
            user_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create the users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check if columns exist (for backward compatibility)
    cursor.execute("PRAGMA table_info(food_items)")
    columns = [info[1] for info in cursor.fetchall()]
    if "track_id" not in columns:
        cursor.execute("ALTER TABLE food_items ADD COLUMN track_id INTEGER")
    if "confidence" not in columns:
        cursor.execute("ALTER TABLE food_items ADD COLUMN confidence REAL")
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE food_items ADD COLUMN user_id INTEGER")
    
    conn.commit()
    conn.close()
    print(f"Banco de dados inicializado em: {DB_PATH}")

def add_food_item(name: str, track_id: int, confidence: float = None, user_id: int = None):
    """Adds a new food item to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO food_items (name, track_id, confidence, user_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            (name, track_id, confidence, user_id, timestamp)
        )
        
        # Obter o ID autoincremento que acabou de ser gerado
        db_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        print(f"Item adicionado ao banco de dados: {name} (User: {user_id}, ID: {db_id}, Confiança: {confidence}) em {timestamp}")
    except Exception as e:
        print(f"Erro ao inserir no banco de dados: {e}")

def get_dashboard_stats(user_id: int):
    """Retrieves statistics for the dashboard for a specific user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Total items
        cursor.execute("SELECT COUNT(*) as total FROM food_items WHERE user_id = ?", (user_id,))
        total_items = cursor.fetchone()['total']
        
        # Avg confidence
        cursor.execute("SELECT AVG(confidence) as avg_conf FROM food_items WHERE user_id = ? AND confidence IS NOT NULL", (user_id,))
        avg_confidence = cursor.fetchone()['avg_conf'] or 0.0
        
        # Counts per category
        cursor.execute("SELECT name, COUNT(*) as count FROM food_items WHERE user_id = ? GROUP BY name ORDER BY count DESC", (user_id,))
        categories = [dict(row) for row in cursor.fetchall()]
        
        # Daily history (last 7 days)
        cursor.execute("""
            SELECT date(timestamp) as day, COUNT(*) as count 
            FROM food_items 
            WHERE user_id = ? AND timestamp >= date('now', '-7 days')
            GROUP BY day 
            ORDER BY day ASC
        """, (user_id,))
        history = [dict(row) for row in cursor.fetchall()]
        
        # Recent activity
        cursor.execute("SELECT name, confidence, timestamp FROM food_items WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5", (user_id,))
        recent = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return {
            "total_items": total_items,
            "avg_confidence": round(avg_confidence * 100, 1),
            "categories": categories,
            "history": history,
            "recent": recent
        }
    except Exception as e:
        print(f"Erro ao buscar estatísticas para usuário {user_id}: {e}")
        return None

def create_user(username: str, password_hash: str):
    """Creates a new user in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    except Exception as e:
        print(f"Erro ao criar usuário: {e}")
        return False

def get_user_by_username(username: str):
    """Retrieves a user by username."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        conn.close()
        if user:
            return {"id": user[0], "username": user[1], "password_hash": user[2]}
        return None
    except Exception as e:
        print(f"Erro ao buscar usuário: {e}")
        return None

# Initialize the database when the module is imported
init_db()
