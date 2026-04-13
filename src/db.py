"""
Database connection and helpers.

Supports:
- PostgreSQL for shared Docker setup
- SQLite fallback for local dev
"""

import os
import ast
import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

# Default to postgres for Docker/shared setup
DB_BACKEND = os.getenv("DB_BACKEND", "postgres").lower()

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    str(Path(__file__).resolve().parent.parent / "db" / "rakuten_colors.db")
)

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rakuten")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rakuten_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rakuten_pass")
DATABASE_URL = os.getenv("DATABASE_URL")

SCHEMA_SQLITE_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
SCHEMA_POSTGRES_PATH = Path(__file__).resolve().parent.parent / "db" / "schema_postgres.sql"


def _postgres_conn():
    import psycopg2

    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)

    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
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
from src.config import DATABASE_PATH

SQLITE_PATH = str(DATABASE_PATH)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@contextmanager
def get_conn():
    conn = None
    try:
        if DB_BACKEND == "postgres":
            conn = _postgres_conn()
        else:
            conn = sqlite3.connect(DATABASE_PATH)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()


def _placeholder():
    return "%s" if DB_BACKEND == "postgres" else "?"


def init_db():
    """
    Create all tables.

    For Postgres in Docker, schema is ideally created automatically by
    docker-entrypoint-initdb.d on first startup. This function remains as
    a fallback/safety net.
    """
    schema_path = SCHEMA_POSTGRES_PATH if DB_BACKEND == "postgres" else SCHEMA_SQLITE_PATH
    schema_sql = schema_path.read_text(encoding="utf-8")

    with get_conn() as conn:
        if DB_BACKEND == "postgres":
            cur = conn.cursor()
            cur.execute(schema_sql)
        else:
            conn.executescript(schema_sql)

    print(f"  DB initialized ({DB_BACKEND})")


def clear_products():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions")
        cur.execute("DELETE FROM labels")
        cur.execute("DELETE FROM products")
    print("  DB cleared (products, labels, predictions)")


def ingest_products(df_x: pd.DataFrame, df_y: pd.DataFrame | None = None, split: str = "train"):
    required_x_cols = {"image_file_name", "item_name", "item_caption"}
    missing_x = required_x_cols - set(df_x.columns)
    if missing_x:
        raise ValueError(f"df_x is missing required columns: {missing_x}")

    if df_y is not None and "color_tags" not in df_y.columns:
        raise ValueError("df_y must contain a 'color_tags' column")

    ph = _placeholder()

    with get_conn() as conn:
        cur = conn.cursor()
        product_ids = []

        for _, row in df_x.iterrows():
            sql = f"""
                INSERT INTO products (split, image_file_name, item_name, item_caption)
                VALUES ({ph}, {ph}, {ph}, {ph})
            """
            cur.execute(
                sql,
                (
                    split,
                    row.get("image_file_name", ""),
                    row.get("item_name", ""),
                    row.get("item_caption", ""),
                ),
            )

            if DB_BACKEND == "postgres":
                cur.execute("SELECT LASTVAL()")
                product_ids.append(cur.fetchone()[0])
            else:
                product_ids.append(cur.lastrowid)
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
                    if DB_BACKEND == "postgres":
                        cur.execute(
                            f"""
                            INSERT INTO labels (product_id, color_tag)
                            VALUES ({ph}, {ph})
                            ON CONFLICT DO NOTHING
                            """,
                            (product_id, tag),
                        )
                    else:
                        cur.execute(
                            f"""
                            INSERT OR IGNORE INTO labels (product_id, color_tag)
                            VALUES ({ph}, {ph})
                            """,
                            (product_id, tag),
                        )
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
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
            return [str(parsed)]
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
    return []


def get_split_data(split: str = "train"):
    with get_conn() as conn:
        df_x = pd.read_sql(
            "SELECT image_file_name, item_name, item_caption FROM products WHERE split = %s"
            if DB_BACKEND == "postgres"
            else "SELECT image_file_name, item_name, item_caption FROM products WHERE split = ?",
            conn,
            params=(split,),
        )

        if DB_BACKEND == "postgres":
            df_y = pd.read_sql(
                """
                SELECT p.id, STRING_AGG(l.color_tag, ',') AS color_tags
                FROM products p
                JOIN labels l ON l.product_id = p.id
                WHERE p.split = %s
                GROUP BY p.id
                """,
                conn,
                params=(split,),
            )
        else:
            df_y = pd.read_sql(
                """
                SELECT p.id, GROUP_CONCAT(l.color_tag) AS color_tags
                FROM products p
                JOIN labels l ON l.product_id = p.id
                WHERE p.split = ?
                GROUP BY p.id
                """,
                conn,
                params=(split,),
            )

    return df_x, df_y


def get_products(split: str = "test"):
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, image_file_name, item_name, item_caption FROM products WHERE split = %s ORDER BY id"
            if DB_BACKEND == "postgres"
            else "SELECT id, image_file_name, item_name, item_caption FROM products WHERE split = ? ORDER BY id",
            conn,
            params=(split,),
# -- Predictions & Runs ------------------------------------------------

def get_split_data(split="train"):
    with get_conn() as conn:
        df_x = pd.read_sql(
            """
            SELECT id AS product_id, image_file_name, item_name, item_caption
            FROM products
            WHERE split = ?
            ORDER BY id
            """,
            conn, params=(split,)
        )
        df_y = pd.read_sql(
            """
            SELECT p.id AS product_id, GROUP_CONCAT(l.color_tag) AS color_tags
            FROM products p
            JOIN labels l ON l.product_id = p.id
            WHERE p.split = ?
            GROUP BY p.id
            ORDER BY p.id
            """,
            conn, params=(split,)
        )
    return df_x, df_y



def get_products(split="test"):
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id AS product_id, image_file_name, item_name, item_caption
            FROM products
            WHERE split = ?
            ORDER BY id
            """,
            conn, params=(split,)
        )
    return df


def save_predictions(product_ids, color_labels, score_matrix, pred_matrix, run_id):
    ph = _placeholder()

def save_predictions(product_ids, color_labels, score_matrix, pred_matrix, run_id):
    """Store all prediction scores for a given run."""
    with get_conn() as conn:
        cur = conn.cursor()
        for pid, scores, preds in zip(product_ids, score_matrix, pred_matrix):
            for j, color in enumerate(color_labels):
                cur.execute(
                    f"""
                    INSERT INTO predictions (product_id, run_id, color_tag, score, predicted)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph})
                    """,
                    (pid, run_id, color, float(scores[j]), bool(preds[j])),
                    "INSERT INTO predictions (product_id, run_id, color_tag, score, predicted) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (pid, run_id, color, float(scores[j]), bool(preds[j]))
                )


def save_run(run_id, model_type, val_f1, params):
    ph = _placeholder()
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_BACKEND == "postgres":
            cur.execute(
                f"""
                INSERT INTO runs (mlflow_run_id, model_type, val_f1, params)
                VALUES ({ph}, {ph}, {ph}, {ph})
                ON CONFLICT DO NOTHING
                """,
                (run_id, model_type, val_f1, json.dumps(params)),
            )
        else:
            cur.execute(
                f"""
                INSERT OR IGNORE INTO runs (mlflow_run_id, model_type, val_f1, params)
                VALUES ({ph}, {ph}, {ph}, {ph})
                """,
                (run_id, model_type, val_f1, json.dumps(params)),
            )
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
    }
