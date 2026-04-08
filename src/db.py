"""
Database connection and helpers.

Uses SQLite locally. PostgreSQL runs in Docker with its own
schema init (db/schema_postgres.sql mounted into the container).
"""
import os
import sqlite3
import ast
import json
import pandas as pd
from pathlib import Path
from contextlib import contextmanager

SQLITE_PATH = os.getenv("SQLITE_PATH",
    str(Path(__file__).resolve().parent.parent / "db" / "rakuten_colors.db"))

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@contextmanager
def get_conn():
    """Return a SQLite connection with WAL mode and foreign keys."""
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables (idempotent)."""
    schema_sql = SCHEMA_PATH.read_text()
    with get_conn() as conn:
        conn.executescript(schema_sql)
    print(f"  DB initialized ({SQLITE_PATH})")


def clear_products():
    """Alle Produkte und abhängige Daten löschen (für Re-Ingest)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM labels")
        conn.execute("DELETE FROM products")
    print("  DB geleert (products, labels, predictions)")


# -- Ingestion --------------------------------------------------------

def ingest_products(df_x, df_y=None, split="train"):
    """
    Insert products + labels into the database.

    Args:
        df_x: DataFrame with [image_file_name, item_name, item_caption]
        df_y: DataFrame with [color_tags] (optional, e.g. test split)
        split: 'train', 'val', 'pseudo_test', 'test'
    """
    with get_conn() as conn:
        cur = conn.cursor()

        product_ids = []
        for _, row in df_x.iterrows():
            cur.execute(
                "INSERT INTO products (split, image_file_name, item_name, item_caption) "
                "VALUES (?, ?, ?, ?)",
                (split, row.get("image_file_name", ""),
                 row.get("item_name", ""), row.get("item_caption", ""))
            )
            product_ids.append(cur.lastrowid)

        label_count = 0
        if df_y is not None:
            for product_id, (_, row) in zip(product_ids, df_y.iterrows()):
                for tag in _parse_color_tags(row["color_tags"]):
                    cur.execute(
                        "INSERT OR IGNORE INTO labels (product_id, color_tag) "
                        "VALUES (?, ?)", (product_id, tag)
                    )
                    label_count += 1

        print(f"  {len(product_ids):>5} products ingested (split={split}, labels={label_count})")
        return product_ids


def _parse_color_tags(raw):
    """Parse color_tags from string repr or list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
    return []


# -- Predictions & Runs ------------------------------------------------

def get_split_data(split="train"):
    with get_conn() as conn:
        df_x = pd.read_sql(
            "SELECT image_file_name, item_name, item_caption FROM products WHERE split = ?",
            conn, params=(split,)
        )
        df_y = pd.read_sql(
            """
            SELECT p.id, GROUP_CONCAT(l.color_tag) as color_tags
            FROM products p JOIN labels l ON l.product_id = p.id
            WHERE p.split = ? GROUP BY p.id
            """,
            conn, params=(split,)
        )
    return df_x, df_y



def get_products(split="test"):
    """Nur Produkte, ohne Labels — für Test-Split."""
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT image_file_name, item_name, item_caption FROM products WHERE split = ?",
            conn, params=(split,)
        )
    return df



def save_predictions(product_ids, color_labels, score_matrix, pred_matrix, run_id):
    """Store all prediction scores for a given run."""
    with get_conn() as conn:
        cur = conn.cursor()
        for pid, scores, preds in zip(product_ids, score_matrix, pred_matrix):
            for j, color in enumerate(color_labels):
                cur.execute(
                    "INSERT INTO predictions (product_id, run_id, color_tag, score, predicted) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (pid, run_id, color, float(scores[j]), bool(preds[j]))
                )


def save_run(run_id, model_type, val_f1, params):
    """Store a training run record."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO runs (mlflow_run_id, model_type, val_f1, params) "
            "VALUES (?, ?, ?, ?)",
            (run_id, model_type, val_f1, json.dumps(params))
        )


# -- Queries -----------------------------------------------------------

def get_product_count(split=None):
    with get_conn() as conn:
        cur = conn.cursor()
        if split:
            cur.execute("SELECT COUNT(*) FROM products WHERE split = ?", (split,))
        else:
            cur.execute("SELECT COUNT(*) FROM products")
        return cur.fetchone()[0]


def get_label_distribution(split=None):
    with get_conn() as conn:
        cur = conn.cursor()
        q = ("SELECT l.color_tag, COUNT(*) as cnt "
             "FROM labels l JOIN products p ON p.id = l.product_id")
        if split:
            cur.execute(q + " WHERE p.split = ? GROUP BY l.color_tag ORDER BY cnt DESC", (split,))
        else:
            cur.execute(q + " GROUP BY l.color_tag ORDER BY cnt DESC")
        return cur.fetchall()


def get_db_summary():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT split, COUNT(*) FROM products GROUP BY split")
        splits = dict(cur.fetchall())
        cur.execute("SELECT COUNT(*) FROM labels")
        n_labels = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM runs")
        n_runs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions")
        n_preds = cur.fetchone()[0]
    return {
        "products_by_split": splits,
        "total_labels": n_labels,
        "total_runs": n_runs,
        "total_predictions": n_preds,
    }
