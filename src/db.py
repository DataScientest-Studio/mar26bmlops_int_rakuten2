# db.py
"""
local: SQLitle (no Docker)
Production: PostgreSQL via DockerCompose

Control via env-variable DB_BACKEND=sqlite|postgres
"""

import os
import sqlite3
import ast                                  # can read python code as string 
from pathlib import Path
from contextlib import contextmanager



# ── config ──────────────────────────────────────────────
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()         # getenv, automatic to Root?

# SQLite
SQLITE_PATH = os.getenv("SQLITE_PATH",
    str(Path(__file__).resolve().parent.parent / "db" / "rakuten_colors.db"))


# PostgreSQL (for Docker)
PG_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),              # to_validate
    "dbname":   os.getenv("DB_NAME", "rakuten_colors"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "secret"),
}

SCHEME_DIR = Path(__file__).resolve().parent.parent / "db"




# ── Connection ─────────────────────────────────────────────────
@contextmanager
def get_conn():
    """Return a DB-Connection (SQLite or PostgreSQL)."""
    if DB_BACKEND == "postgres":
        import psycopg2
        conn = psycopg2.connect(**PG_CONFIG)
    else:
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
    """Create all tables)."""
    if DB_BACKEND == "postgres":
        schema_file = SCHEME_DIR / "schema_postgres.sql"
    else:
        schema_file = SCHEME_DIR / "schema.sql"

    schema_sql = schema_file.read_text()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.executescript(schema_sql) if DB_BACKEND == "sqlite" \
            else cur.execute(schema_sql)

    print(f"DB initialized ({DB_BACKEND}: {SQLITE_PATH if DB_BACKEND == 'sqlite' else PG_CONFIG['host']})")


# ── read data ─────────────────────────────────────────────
def ingest_products(df_x, df_y=None, split="train"):
    """
    Lädt Produkte + Labels in die Datenbank.

    Args:
        df_x: DataFrame mit Spalten [image_file_name, item_name, item_caption]
        df_y: DataFrame mit Spalte [color_tags] (optional, z.B. für Test-Split)
        split: 'train', 'val', 'pseudo_test', 'test'
    """
    with get_conn() as conn:
        cur = conn.cursor()

        # ── Produkte einfügen ──
        product_ids = []
        for _, row in df_x.iterrows():
            cur.execute(
                """INSERT INTO products (split, image_file, item_name, item_caption)
                   VALUES (?, ?, ?, ?)""",
                (
                    split,
                    row.get("image_file_name", ""),
                    row.get("item_name", ""),
                    row.get("item_caption", ""),
                )
            )
            product_ids.append(cur.lastrowid)

        # ── Labels einfügen (falls vorhanden) ──
        label_count = 0
        if df_y is not None:
            for product_id, (_, row) in zip(product_ids, df_y.iterrows()):
                tags = _parse_color_tags(row["color_tags"])
                for tag in tags:
                    cur.execute(
                        """INSERT OR IGNORE INTO labels (product_id, color_tag)
                           VALUES (?, ?)""",
                        (product_id, tag)
                    )
                    label_count += 1

        print(f" {len(product_ids):>5} products loaded (split={split}, labels={label_count})")
        return product_ids




def _parse_color_tags(raw):
    """Parst color_tags – egal ob String-Repr oder schon Liste."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
    return []



def get_db_summary():
    """Return DB content"""
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