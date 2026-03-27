PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    split TEXT NOT NULL,
    image_file_name TEXT,
    item_name TEXT,
    item_caption TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    color_tag TEXT NOT NULL,
    UNIQUE(product_id, color_tag),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mlflow_run_id TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    val_f1 REAL,
    params TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    color_tag TEXT NOT NULL,
    score REAL,
    predicted BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_products_split
ON products(split);

CREATE INDEX IF NOT EXISTS idx_labels_product_id
ON labels(product_id);

CREATE INDEX IF NOT EXISTS idx_labels_color_tag
ON labels(color_tag);

CREATE INDEX IF NOT EXISTS idx_predictions_product_id
ON predictions(product_id);

CREATE INDEX IF NOT EXISTS idx_predictions_run_id
ON predictions(run_id);
