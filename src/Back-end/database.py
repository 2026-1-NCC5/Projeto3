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
            weight_kg REAL DEFAULT 0.0,
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

    # Pending camera batch per active camera session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS camera_pending_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            weight_kg REAL NOT NULL DEFAULT 0,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check if columns exist (for backward compatibility)
    cursor.execute("PRAGMA table_info(food_items)")
    columns = [info[1] for info in cursor.fetchall()]
    if "track_id" not in columns:
        cursor.execute("ALTER TABLE food_items ADD COLUMN track_id INTEGER")
    if "weight_kg" not in columns:
        cursor.execute("ALTER TABLE food_items ADD COLUMN weight_kg REAL DEFAULT 0.0")
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE food_items ADD COLUMN user_id INTEGER")

    cursor.execute("PRAGMA table_info(camera_pending_items)")
    pending_columns = [info[1] for info in cursor.fetchall()]
    if not pending_columns:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_pending_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 1,
                weight_kg REAL NOT NULL DEFAULT 0,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    conn.commit()
    conn.close()
    print(f"Banco de dados inicializado em: {DB_PATH}")

def add_food_item(name: str, track_id: int, confidence: float = None, user_id: int = None, weight_kg: float = 0.0):
    """Adds a new food item to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO food_items (name, track_id, confidence, weight_kg, user_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (name, track_id, confidence, weight_kg, user_id, timestamp)
        )
        
        # Obter o ID autoincremento que acabou de ser gerado
        db_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        print(f"Item adicionado ao banco de dados: {name} (User: {user_id}, ID: {db_id}, Confiança: {confidence}) em {timestamp}")
        return db_id
    except Exception as e:
        print(f"Erro ao adicionar item ao banco de dados: {e}")
        return None

def get_all_food_items(user_id: int):
    """Retrieves all food items for a specific user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM food_items WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Erro ao buscar itens: {e}")
        return []

def update_food_item(item_id: int, name: str, weight_kg: float, confidence: float):
    """Updates an existing food item."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE food_items SET name = ?, weight_kg = ?, confidence = ? WHERE id = ?",
            (name, weight_kg, confidence, item_id)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except Exception as e:
        print(f"Erro ao atualizar item {item_id}: {e}")
        return False

def delete_food_item(item_id: int):
    """Deletes a food item from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM food_items WHERE id = ?", (item_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except Exception as e:
        print(f"Erro ao deletar item {item_id}: {e}")
        return False
        print(f"Erro ao inserir no banco de dados: {e}")


def add_pending_camera_item(session_key: str, user_id: int, name: str, quantity: int = 1, weight_kg: float = 0.0, confidence: float = None):
    """Adds or increments a pending camera detection for a camera session."""
    if not session_key or not user_id or not name:
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, quantity, weight_kg, confidence FROM camera_pending_items WHERE session_key = ? AND user_id = ? AND name = ?",
            (session_key, user_id, name),
        )
        row = cursor.fetchone()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if row:
            pending_id, current_quantity, current_weight, current_confidence = row
            new_quantity = int(current_quantity or 0) + int(quantity)
            new_weight = float(current_weight or 0.0) + float(weight_kg or 0.0)
            new_confidence = confidence if confidence is not None else current_confidence
            cursor.execute(
                """
                UPDATE camera_pending_items
                SET quantity = ?, weight_kg = ?, confidence = ?, timestamp = ?
                WHERE id = ?
                """,
                (new_quantity, new_weight, new_confidence, timestamp, pending_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO camera_pending_items (session_key, user_id, name, quantity, weight_kg, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_key, user_id, name, quantity, weight_kg, confidence, timestamp),
            )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao inserir item pendente da câmera: {e}")


def get_pending_camera_batch(session_key: str, user_id: int):
    """Returns pending camera batch grouped by food name."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name, SUM(quantity) as quantity, SUM(weight_kg) as weight_kg, MAX(confidence) as confidence, MAX(timestamp) as last_seen
            FROM camera_pending_items
            WHERE session_key = ? AND user_id = ?
            GROUP BY name
            ORDER BY last_seen DESC
            """,
            (session_key, user_id),
        )
        items = [dict(row) for row in cursor.fetchall()]

        total_quantity = sum(int(item.get("quantity") or 0) for item in items)
        total_weight_kg = round(sum(float(item.get("weight_kg") or 0.0) for item in items), 2)

        conn.close()
        return {
            "items": items,
            "total_quantity": total_quantity,
            "total_weight_kg": total_weight_kg,
        }
    except Exception as e:
        print(f"Erro ao buscar lote pendente da câmera: {e}")
        return {"items": [], "total_quantity": 0, "total_weight_kg": 0.0}


def delete_pending_camera_item(session_key: str, user_id: int, name: str):
    """Deletes all entries of a specific food item from a pending camera session."""
    if not session_key or not user_id or not name:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM camera_pending_items WHERE session_key = ? AND user_id = ? AND name = ?",
            (session_key, user_id, name),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Erro ao deletar item pendente: {e}")
        return False


def clear_pending_camera_batch(session_key: str, user_id: int = None):
    """Clears pending camera detections for a session."""
    if not session_key:
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if user_id is None:
            cursor.execute("DELETE FROM camera_pending_items WHERE session_key = ?", (session_key,))
        else:
            cursor.execute("DELETE FROM camera_pending_items WHERE session_key = ? AND user_id = ?", (session_key, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao limpar lote pendente da câmera: {e}")

def get_dashboard_stats(user_id: int, name_filter: str = None, date_from: str = None):
    """Retrieves statistics for the dashboard for a specific user with optional filters."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Base queries
        where_clause = "WHERE user_id = ?"
        params = [user_id]
        
        if name_filter:
            where_clause += " AND name LIKE ?"
            params.append(f"%{name_filter}%")
        
        if date_from:
            where_clause += " AND timestamp >= ?"
            params.append(date_from)

        # Total items
        cursor.execute(f"SELECT COUNT(*) as total FROM food_items {where_clause}", params)
        total_items = cursor.fetchone()['total']
        
        # Avg confidence
        cursor.execute(f"SELECT AVG(confidence) as avg_conf FROM food_items {where_clause} AND confidence IS NOT NULL", params)
        avg_confidence = cursor.fetchone()['avg_conf'] or 0.0

        # Total weight
        cursor.execute(f"SELECT SUM(weight_kg) as total_weight FROM food_items {where_clause}", params)
        total_weight = cursor.fetchone()['total_weight'] or 0.0
        
        # Counts per category
        cursor.execute(f"SELECT name, COUNT(*) as count FROM food_items {where_clause} GROUP BY name ORDER BY count DESC", params)
        categories = [dict(row) for row in cursor.fetchall()]
        
        # Daily history (last 7-30 days depending on filter)
        history_limit = "date('now', '-7 days')" if not date_from else "?"
        history_params = params + ([date_from] if date_from else [])
        
        cursor.execute(f"""
            SELECT date(timestamp) as day, COUNT(*) as count 
            FROM food_items 
            {where_clause} AND timestamp >= {history_limit}
            GROUP BY day 
            ORDER BY day ASC
        """, history_params)
        history = [dict(row) for row in cursor.fetchall()]
        
        # Recent activity
        cursor.execute(f"SELECT name, confidence, weight_kg, timestamp FROM food_items {where_clause} ORDER BY timestamp DESC LIMIT 5", params)
        recent = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return {
            "total_items": total_items,
            "total_weight": round(total_weight, 1),
            "avg_confidence": round(avg_confidence * 100, 1),
            "categories": categories,
            "history": history,
            "recent": recent
        }
    except Exception as e:
        print(f"Erro ao buscar estatísticas para usuário {user_id}: {e}")
        return None
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

def update_user(user_id: int, username: str = None, password_hash: str = None):
    """Updates user information."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if username and password_hash:
            cursor.execute("UPDATE users SET username = ?, password_hash = ? WHERE id = ?", (username, password_hash, user_id))
        elif username:
            cursor.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
        elif password_hash:
            cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Erro ao atualizar usuário: {e}")
        return False

# Initialize the database when the module is imported
init_db()
