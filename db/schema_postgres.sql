CREATE TABLE IF NOT EXISTS products (
    id              SERIAL PRIMARY KEY,
    split           VARCHAR(20) NOT NULL,
    image_file_name TEXT NOT NULL,
    item_name       TEXT,
    item_caption    TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    id              SERIAL PRIMARY KEY,
    product_id      INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    color_tag       VARCHAR(50) NOT NULL,
    UNIQUE(product_id, color_tag)
);

CREATE TABLE IF NOT EXISTS runs (
    id              SERIAL PRIMARY KEY,
    mlflow_run_id   TEXT UNIQUE NOT NULL,
    model_type      VARCHAR(100),
    val_f1          FLOAT,
    params          JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    product_id      INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    run_id          TEXT NOT NULL,
    color_tag       VARCHAR(50) NOT NULL,
    score           DOUBLE PRECISION,
    predicted       BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_products_split ON products(split);
CREATE INDEX IF NOT EXISTS idx_labels_product ON labels(product_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_product ON predictions(product_id);

CREATE OR REPLACE VIEW predictions_with_image AS
SELECT
    pr.id              AS prediction_id,
    pr.run_id,
    pr.color_tag,
    pr.score,
    pr.predicted,
    p.id               AS product_id,
    p.image_file_name,
    p.item_name,
    p.item_caption,
    p.split
FROM predictions pr
JOIN products p ON p.id = pr.product_id;