-- ============================================================
-- Rakuten Color Extraction – PostgreSQL Schema
-- Wird beim docker-compose up automatisch ausgeführt
-- ============================================================

CREATE TABLE IF NOT EXISTS products (
    id              SERIAL PRIMARY KEY,
    split           VARCHAR(20) NOT NULL,
    image_file_name TEXT,
    item_name       TEXT,
    item_caption    TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS labels (
    id              SERIAL PRIMARY KEY,
    product_id      INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    color_tag       VARCHAR(50) NOT NULL,
    UNIQUE(product_id, color_tag)
);

CREATE TABLE IF NOT EXISTS runs (
    id              SERIAL PRIMARY KEY,
    mlflow_run_id   TEXT UNIQUE,
    model_type      VARCHAR(50),
    val_f1          FLOAT,
    params          JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    product_id      INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    run_id          TEXT NOT NULL,
    color_tag       VARCHAR(50) NOT NULL,
    score           FLOAT,
    predicted       BOOLEAN
);

-- Indices für schnelle Abfragen
CREATE INDEX IF NOT EXISTS idx_products_split ON products(split);
CREATE INDEX IF NOT EXISTS idx_labels_product ON labels(product_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_product ON predictions(product_id);

-- Prediction Overview with img file
CREATE OR REPLACE VIEW predictions_with_image AS
SELECT
    p.id              AS prediction_id,
    p.run_id,
    p.color_tag,
    p.score,
    p.predicted,
    pr.id             AS product_id,
    pr.image_file_name,
    pr.split,
    pr.item_name
FROM predictions p
JOIN products pr ON pr.id = p.product_id;
