"""Storage layer. SQLite is used here for local testing; the same SQL runs on MySQL in deployment."""
import os, sqlite3
import pandas as pd

DB_PATH = os.environ.get("FOODWASTE_DB", "foodwaste.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS food_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT UNIQUE NOT NULL,
    unit_cost REAL NOT NULL,
    price REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES food_items(item_id),
    sale_date TEXT NOT NULL,
    prepared_qty INTEGER NOT NULL,
    sold_qty INTEGER NOT NULL,
    waste_qty INTEGER NOT NULL,
    UNIQUE(item_id, sale_date));
CREATE TABLE IF NOT EXISTS inventory (
    stock_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES food_items(item_id),
    stock_date TEXT NOT NULL,
    opening_stock INTEGER NOT NULL,
    closing_stock INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES food_items(item_id),
    target_date TEXT NOT NULL,
    planned_qty INTEGER NOT NULL,
    predicted_waste REAL NOT NULL,
    predicted_demand REAL NOT NULL,
    recommended_qty INTEGER NOT NULL,
    risk_level TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""

def connect():
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    return con

def init_db(csv_path="data/food_waste.csv"):
    con = connect(); con.executescript(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM food_items").fetchone()[0] == 0 and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for (name, cost, price) in df[["item", "unit_cost", "price"]].drop_duplicates().itertuples(index=False):
            con.execute("INSERT INTO food_items(item_name, unit_cost, price) VALUES (?,?,?)", (name, cost, price))
        ids = {r["item_name"]: r["item_id"] for r in con.execute("SELECT * FROM food_items")}
        con.executemany("INSERT OR IGNORE INTO sales(item_id, sale_date, prepared_qty, sold_qty, waste_qty) VALUES (?,?,?,?,?)",
                        [(ids[r.item], r.date, int(r.prepared_qty), int(r.sold_qty), int(r.waste_qty))
                         for r in df.itertuples(index=False)])
    con.commit(); con.close()
