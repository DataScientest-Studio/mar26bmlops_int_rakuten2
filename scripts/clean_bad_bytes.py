"""
One-time DB cleanup: find products with invalid UTF-8 bytes in item_name or
item_caption and replace those bytes with '?'.

Why this exists:
    The Rakuten source CSV contains a small number of rows with non-UTF-8
    bytes (Windows-1252 leftovers). When pandas / psycopg2 try to read
    these rows, they crash with UnicodeDecodeError inside psycopg2's
    C-extension — and every runtime workaround we've tried gets bypassed
    by the C-extension.

    Fix: clean the data ONCE at rest, then all downstream reads work
    normally with the default UTF-8 client encoding.

How it works:
    1. Use COPY TO STDOUT as raw bytes (COPY bypasses psycopg2's text
       type casters entirely).
    2. Parse the bytes ourselves with errors='replace' to build a list
       of (id, safe_name, safe_caption) tuples.
    3. Only UPDATE rows that actually changed.

Run this ONCE from inside the training container (or any container with
psycopg2 + access to the rakuten DB):

    docker run --rm --network rakuten2_default \\
      -v /home/mirco/rakuten2/src:/app/src \\
      -v /home/mirco/rakuten2/scripts:/app/scripts \\
      -w /app \\
      -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/rakuten \\
      rakuten2-training \\
      python /app/scripts/clean_bad_bytes.py

      
      
NETWORK=$(docker network ls --format '{{.Name}}' | grep rakuten | head -1)

docker run --rm --network "$NETWORK" \
  -v /home/mirco/rakuten2/src:/app/src \
  -v /home/mirco/rakuten2/scripts:/app/scripts \
  -w /app \
  -e DATABASE_URL=postgresql://postgres:postgres@postgres:5432/rakuten \
  rakuten2-training \
  python /app/scripts/clean_bad_bytes.py      
      
      
      
      
"""

import os
from io import BytesIO

import psycopg2

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/rakuten",
)


def main():
    conn = psycopg2.connect(DATABASE_URL)

    # Force raw-byte passthrough for the COPY stream
    conn.set_client_encoding("SQL_ASCII")

    cur = conn.cursor()

    # -- Step 1: pull all (id, item_name, item_caption) as raw CSV bytes --
    print("Exporting products via COPY TO STDOUT ...")
    buf = BytesIO()
    cur.copy_expert(
        """
        COPY (
            SELECT id, COALESCE(item_name, ''), COALESCE(item_caption, '')
            FROM products
            ORDER BY id
        )
        TO STDOUT
        WITH (FORMAT csv, DELIMITER E'\\x01', QUOTE E'\\x02')
        """,
        buf,
    )
    raw_csv = buf.getvalue()
    print(f"  Got {len(raw_csv):,} raw bytes")

    # -- Step 2: split lines and decode each safely --
    # We use byte-level parsing because some lines may contain non-UTF-8 bytes
    lines = raw_csv.split(b"\n")
    # Drop empty last line from trailing newline
    if lines and lines[-1] == b"":
        lines.pop()

    print(f"  Parsed {len(lines):,} lines")

    # -- Step 3: find rows with invalid UTF-8, collect fixes --
    # DELIMITER is \x01 (SOH), QUOTE is \x02 (STX) — both very unlikely
    # to appear in product descriptions
    updates = []
    bad_byte_count = 0

    for line_no, raw_line in enumerate(lines):
        # Split on \x01 delimiter into 3 fields: id, name, caption
        parts = raw_line.split(b"\x01")
        if len(parts) != 3:
            # unexpected line, skip (e.g. embedded newline in caption)
            continue

        id_bytes, name_bytes, caption_bytes = parts

        # Strip QUOTE characters if present
        name_bytes = name_bytes.strip(b"\x02")
        caption_bytes = caption_bytes.strip(b"\x02")

        try:
            id_str = id_bytes.decode("ascii")
            product_id = int(id_str)
        except (UnicodeDecodeError, ValueError):
            print(f"  WARN: skipping unparseable id on line {line_no}")
            continue

        # Try strict UTF-8 first; only flag as "bad" if strict fails
        name_str = None
        caption_str = None
        has_bad = False

        try:
            name_str = name_bytes.decode("utf-8")
        except UnicodeDecodeError:
            name_str = name_bytes.decode("utf-8", errors="replace")
            has_bad = True

        try:
            caption_str = caption_bytes.decode("utf-8")
        except UnicodeDecodeError:
            caption_str = caption_bytes.decode("utf-8", errors="replace")
            has_bad = True

        if has_bad:
            bad_byte_count += 1
            updates.append((name_str, caption_str, product_id))

    print(f"  Found {bad_byte_count} rows with invalid UTF-8 bytes")

    if not updates:
        print("DB is already clean. Nothing to do.")
        cur.close()
        conn.close()
        return

    # -- Step 4: reset to UTF-8 for the UPDATE statements --
    conn.set_client_encoding("UTF8")
    cur = conn.cursor()

    # Batch update — psycopg2 executemany is fine for ~hundreds of rows
    print(f"Updating {len(updates)} rows ...")
    cur.executemany(
        "UPDATE products SET item_name = %s, item_caption = %s WHERE id = %s",
        updates,
    )
    conn.commit()
    print(f"  Done. {cur.rowcount} rows updated.")

    cur.close()
    conn.close()

    # -- Step 5: verify --
    print("\nVerifying ...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM products")
    total = cur.fetchone()[0]
    print(f"  Products table: {total:,} rows")

    # Try reading everything — if any bad bytes remain, this would still fail.
    # But we only try a sample to keep it quick.
    cur.execute(
        "SELECT item_name, item_caption FROM products "
        "WHERE split = 'val' LIMIT 100"
    )
    try:
        rows = cur.fetchall()
        print(f"  Successfully read first 100 val rows, no decode error.")
    except UnicodeDecodeError as e:
        print(f"  WARN: still getting decode errors: {e}")
        print(f"  You may need to re-run this script or investigate further.")

    cur.close()
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()