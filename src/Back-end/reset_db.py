import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "food_inventory.db"

def reset_database():
    """Apaga todos os dados da tabela e reseta o ID autoincremento para voltar a 1."""
    if not DB_PATH.exists():
        print("O banco de dados ainda não existe.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Apaga todos os registros
        cursor.execute("DELETE FROM food_items")
        cursor.execute("DELETE FROM users")
        
        # Reseta os contadores de ID (autoincremento)
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='food_items'")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='users'")
        
        conn.commit()
        conn.close()
        print("Banco de dados resetado com sucesso! Tabelas de alimentos e usuários limpas.")
    except Exception as e:
        print(f"Erro ao resetar o banco de dados: {e}")

if __name__ == "__main__":
    reset_database()
