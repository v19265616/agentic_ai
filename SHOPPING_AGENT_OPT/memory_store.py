import os
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "store.db"
print("MEMORY DATABASE:", DB_PATH)
USER_ID = os.getenv("SHOPPING_USER_ID", "demo_user")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            prefers_organic INTEGER,
            max_price REAL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def get_preferences():
    conn = _connect()
    row = conn.execute(
        "SELECT prefers_organic, max_price, updated_at "
        "FROM user_preferences WHERE user_id = ?",
        (USER_ID,),
    ).fetchone()
    conn.close()
    if row is None:
        return {"prefers_organic": None, "max_price": None}
    return {
        "prefers_organic": None if row["prefers_organic"] is None
        else bool(row["prefers_organic"]),
        "max_price": row["max_price"],
        "updated_at": row["updated_at"],
    }


def save_preferences(prefers_organic=None, max_price=None):
    current = get_preferences()
    organic = current["prefers_organic"] if prefers_organic is None else prefers_organic
    price = current["max_price"] if max_price is None else max_price

    conn = _connect()
    conn.execute("""
        INSERT INTO user_preferences
            (user_id, prefers_organic, max_price, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(user_id) DO UPDATE SET
            prefers_organic = excluded.prefers_organic,
            max_price = excluded.max_price,
            updated_at = datetime('now')
    """, (USER_ID, None if organic is None else int(organic), price))
    conn.commit()
    conn.close()
    return get_preferences()

def get_orders(limit=20):
    conn = _connect()

    rows = conn.execute("""
        SELECT id, product_id, product_name, price, ordered_at
        FROM orders
        ORDER BY datetime(ordered_at) DESC, id DESC
        LIMIT ?
    """, (max(1, min(int(limit), 100)),)).fetchall()

    conn.close()

    orders = [dict(row) for row in rows]

    print("ORDERS FETCHED FROM SQLITE:", orders)

    return orders
