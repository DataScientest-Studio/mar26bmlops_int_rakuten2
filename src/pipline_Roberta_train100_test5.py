"""
TEMP TEST PIPELINE

Trainiert XLM-RoBERTa auf 100 echten Samples
und führt danach eine echte 5er-Test-Inference durch.

Use:
    python -m src.pipeline_test --mode full
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    COLOR_LABELS, DATA_DIR, XLM_CONFIG, MODEL_DIR
)

from src.db import init_db, ingest_products, get_db_summary
from src.models.train_model import train_xlm, TextColorClassifier


# ───────────────────────────────────────────────────────────────
# 5‑Sample REAL Inference
# ───────────────────────────────────────────────────────────────
def run_real_test_inference(df_test):
    print("\n[TEST] Lade bestes Modell…")

    tokenizer = AutoTokenizer.from_pretrained(XLM_CONFIG["model_name"])
    model_path = MODEL_DIR / "xlm_best.pt"

    model = TextColorClassifier(XLM_CONFIG["model_name"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Nimm 5 echte Samples
    df_small = df_test.head(5)

    print("\n[TEST] Predictions für 5 echte Test-Samples:\n")

    for idx, row in df_small.iterrows():
        text = f"{row['item_name']} [SEP] {row.get('item_caption', '')}"

        enc = tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits).numpy()[0]

        print(f"\n--- SAMPLE {idx} ---")
        print("TEXT:", text[:200], "...")
        print("PREDICTIONS:")
        for color, p in zip(COLOR_LABELS, probs):
            if p > 0.15:
                print(f"  {color:15s} {p:.3f}")

    print("\n[TEST] Fertig.\n")


# ───────────────────────────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────────────────────────
def run_pipeline(mode="full"):
    print("=" * 60)
    print("RAKUTEN COLOR EXTRACTION PIPELINE (TEMP TEST VERSION)")
    print(f"  Mode: {mode}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/X] Data loading...")
    df_x    = pd.read_csv(DATA_DIR / "raw" / "X_train.csv")
    df_y    = pd.read_csv(DATA_DIR / "raw" / "y_train.csv")
    df_test = pd.read_csv(DATA_DIR / "raw" / "X_test.csv")

    print(f"  Train: {len(df_x)}, Test: {len(df_test)}")

    # 2. Splits
    print("\n[2/X] Train/Val/Pseudo-Test Split...")
    train_x, temp_x, train_y, temp_y = train_test_split(
        df_x, df_y, test_size=0.2, random_state=42
    )
    val_x, pseudo_x, val_y, pseudo_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=42
    )
    print(f"  Train={len(train_x)}, Val={len(val_x)}, Pseudo={len(pseudo_x)}")

    # 3. DB ingest
    if mode in ("full", "train"):
        print("\n[3/X] Datenbank befüllen…")
        init_db()
        ingest_products(train_x, train_y,   split="train")
        ingest_products(val_x,   val_y,     split="val")
        ingest_products(pseudo_x, pseudo_y, split="pseudo_test")
        ingest_products(df_test, df_y=None, split="test")

        summary = get_db_summary()
        print(f"  DB: {summary['products_by_split']}")

    if mode == "ingest":
        print("\nReady (Ingest only).")
        return

    # 4. Mini‑Training (100 echte Samples)
    if mode in ("full", "train"):
        print("\n[4/X] XLM-RoBERTa Training auf 100 echten Samples…")

        mini_config = {
            **XLM_CONFIG,
            "epochs": 1,
            "batch_size": 8,
            "lr": 5e-5,
            "max_len": 128
        }

        train_xlm(
            train_x.head(100), train_y.head(100),
            val_x.head(20),    val_y.head(20),
            mini_config
        )

    # 5. REAL Inference (5 echte Samples)
    if mode in ("full", "predict", "train"):
        run_real_test_inference(df_test)

    print("\nPipeline fertig.\n")


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(mode="full")


# 'output:'


# --- SAMPLE 0 ---
# TEXT: KYV39 miraie f ミライエ フォルテ kyv39 au エーユー スマホ ケース スマホカバー PC ハードケース リボン　レース　ストライプ ラブリー 004927 [SEP] 注意事項 素材ケースの入荷時期により、商品写真とは一部カメラ・ボタン穴などの形状が異なるケースを使用する場合があります。また、一部機種のTPU商品は素材強度の関係でケースの上からボタン等を操作するタイプとなり ...
# PREDICTIONS:
#   Black           0.646
#   White           0.548
#   Grey            0.341
#   Navy            0.267
#   Blue            0.259
#   Red             0.397
#   Pink            0.339
#   Brown           0.325
#   Beige           0.305
#   Green           0.223
#   Khaki           0.253
#   Orange          0.168
#   Purple          0.212
#   Gold            0.159
#   Silver          0.165
#   Transparent     0.170
#   Multiple Colors 0.232

# --- SAMPLE 1 ---
# TEXT: SO-04K Xperia XZ2 Premium エクスペリア エックスゼットツー プレミアム docomo so04k ドコモ スマホ カバー スマホケース スマホカバー PC ハードケース 011123 カメラ ビデオ 写真 [SEP] 注意事項素材ケースの入荷時期により、商品写真とは一部カメラ・ボタン穴などの形状が異なるケースを使用する場合があります。また、一部機種のTPU商品は素材強度の ...
# PREDICTIONS:
#   Black           0.643
#   White           0.549
#   Grey            0.342
#   Navy            0.271
#   Blue            0.265
#   Red             0.395
#   Pink            0.340
#   Brown           0.321
#   Beige           0.300
#   Green           0.217
#   Khaki           0.254
#   Orange          0.166
#   Purple          0.210
#   Gold            0.160
#   Silver          0.162
#   Transparent     0.170
#   Multiple Colors 0.234

# --- SAMPLE 2 ---
# TEXT: MO-01K MONO モノ mo01k docomo ドコモ 手帳型 スマホ カバー カバー レザー ケース 手帳タイプ フリップ ダイアリー 二つ折り 革 その他 キャラクター　模様 004381 [SEP] 商品特徴 ・シームレスな全面デザイン（内側は落ち着いたオフホワイト） ・外側は高品質の人工革を使用 ・内側にはスマホを装着するケースが付属 ・機種専用のカメラホール仕様でケースを装着した ...
# PREDICTIONS:
#   Black           0.669
#   White           0.532
#   Grey            0.320
#   Navy            0.277
#   Blue            0.227
#   Red             0.391
#   Pink            0.357
#   Brown           0.304
#   Beige           0.299
#   Green           0.188
#   Khaki           0.215
#   Orange          0.151
#   Purple          0.201
#   Transparent     0.156
#   Multiple Colors 0.210

# --- SAMPLE 3 ---
# TEXT: Xperia XZ 手帳型ケース ビーチ ハワイ エクスペリア ケース カバー ケイオー ブランド 手帳型 全機種対応 全面保護 カード収納 高品質 PU レザー スマホケース スマホカバー マグネット 薄型 カード入れ 携帯ケース 携帯カバー [SEP] 対応機種Xperia XZ ( エクスペリア ) ソニー対応型番XperiaXZキャリアDoCoMo ドコモ au SoftBank ソフトバ ...
# PREDICTIONS:
#   Black           0.646
#   White           0.510
#   Grey            0.299
#   Navy            0.273
#   Blue            0.248
#   Red             0.392
#   Pink            0.362
#   Brown           0.293
#   Beige           0.288
#   Green           0.190
#   Khaki           0.239
#   Orange          0.156
#   Purple          0.199
#   Gold            0.159
#   Silver          0.166
#   Multiple Colors 0.227

# --- SAMPLE 4 ---
# TEXT: 【中古】コムサデモード COMME CA DU MODE スカート ボムトス ロング丈 斜めストライプ バックスリット コットン グレー 白 9 レディース 【ベクトル 古着】 200315 ベクトルプレミアム店 [SEP] 【中古】コムサデモード COMME CA DU MODE スカート ボムトス ロング丈 斜めストライプ バックスリット コットン グレー 白 9 レディース 【ベクトル 古着 ...
# PREDICTIONS:
#   Black           0.680
#   White           0.498
#   Grey            0.298
#   Navy            0.292
#   Blue            0.236
#   Red             0.379
#   Pink            0.354
#   Brown           0.308
#   Beige           0.312
#   Green           0.186
#   Khaki           0.232
#   Orange          0.154
#   Purple          0.202
#   Gold            0.150
#   Silver          0.156
#   Transparent     0.155
#   Multiple Colors 0.204

# [TEST] Fertig.


# Pipeline fertig.

# (venv) mirco@wsl:~/rakuten2$ 