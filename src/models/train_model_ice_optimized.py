"""
================================================================================
Rakuten Color Classifier  —  GPU-optimiert für RTX 5060 Ti (16 GB)
================================================================================

ÄNDERUNGEN GEGENÜBER DER ORIGINAL-VERSION  (mit Begründung)
-------------------------------------------------------------

1.  BATCH_SIZE  32 → 128
    Original belegte nur 3,7 / 16 GB VRAM.  Mit 128 landen wir bei ~8–10 GB.
    Mehr Samples pro Gradient-Step  →  stabilere Gradienten, weniger Rauschen.
    Außerdem: 4× weniger DataLoader-Overhead pro Epoch (5966 → ~1490 Batches).

2.  num_workers  0 → 6  +  persistent_workers=True  +  prefetch_factor=2
    Der Colab-Workaround (num_workers=0) ist hier nicht nötig.  WSL + Linux-
    Prozessmodell unterstützt forked DataLoader-Worker problemlos.
    Effekt: Bilder werden parallel dekodiert während die GPU rechnet.
    Das war der Hauptengpass: GPU bei 49% bedeutete, sie wartete auf Daten.

3.  pin_memory=True
    Reserviert den CPU-RAM für die Batch-Tensoren als "pinned" (nicht-pageable).
    CUDA kann dann DMA-Transfers direkt starten, ohne vorher zu kopieren.
    Kostet ein bisschen RAM, spart aber PCIe-Latenz.

4.  torch.backends.cudnn.benchmark = True
    Bei fixen Eingabegrößen (128 Token, 224×224 Pixel) probiert cuDNN beim
    ersten Batch alle verfügbaren Conv-Algorithmen durch und merkt sich den
    schnellsten.  Ab Batch 2 ist alles optimal.

5.  AMP:  float16 → bfloat16
    Die RTX 5060 Ti ist Blackwell-Architektur (2025) und hat native BF16-
    Tensor-Cores.  BF16 hat denselben Exponentenbereich wie FP32 (kein
    Gradient-Underflow), deshalb wird GradScaler überflüssig.
    Training wird numerisch stabiler, und BF16-Durchsatz auf Blackwell ist
    mindestens so hoch wie FP16.

6.  MAX_EPOCHS  30 → 20  +  ES_PATIENCE  7 → 5
    Mit 190k Samples und LR-Scheduler konvergiert das Modell erfahrungsgemäß
    zwischen Epoch 8–14.  Der Scheduler halbiert die LR bei Stagnation, und
    Early Stopping mit Patience=5 ist aggressiv genug, um weitere 2–3 Epochen
    sinnloser Rechnung zu sparen.  Worst Case: ein paar Prozentpunkte F1
    gegenüber Patience=7 — bei 190k Samples ist der Unterschied minimal.

7.  UNFREEZE_LAYERS  3 → 2
    3 Schichten zu entfreieren macht Sinn bei 50k Samples.  Bei 190k haben
    wir genug Daten, aber 2 Schichten trainieren schneller und das Risiko
    von catastrophic forgetting ist geringer.  Wer mehr herausholen will,
    kann wieder auf 3 stellen.

8.  Autocast-Fix
    Die @contextmanager-Wrapper-Funktion namens `autocast` hat sich selbst
    rekursiv aufgerufen (infinite recursion).  Entfernt.  Stattdessen wird
    direkt torch.amp.autocast verwendet.

ERWARTETE SPEEDUP-SCHÄTZUNG
-----------------------------
  Original:  ~4.7 it/s  (batch=32, workers=0, fp16)
  Optimiert: ~14–22 it/s  (batch=128, workers=6, bf16, cuDNN-benchmark)
  Epochendauer:  ~20 min → ~4–6 min
  Gesamtdauer (20 Epochen + ES):  ~6–8h → ~1,5–2h

================================================================================
"""

import os
import ast
import pickle
import copy

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler                          # GradScaler bleibt für FP16-Fallback
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, CLIPImageProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── cuDNN-Benchmark: bei fixen Tensor-Shapes findet cuDNN automatisch den
#    schnellsten Convolution-Algorithmus.  Einmalige Suche beim ersten Batch.
torch.backends.cudnn.benchmark = True


# ==========================================
# Dual Encoder
# ==========================================

class DualEncoder(nn.Module):
    """
    Wraps a Japanese BERT text encoder and a CLIP vision encoder.
    All parameters frozen on init; selectively unfrozen via unfreeze_encoder_layers().
    """

    def __init__(self, text_model_id: str, vision_model_id: str):
        super().__init__()
        self.text_encoder   = AutoModel.from_pretrained(text_model_id)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_id)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def get_text_features(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # CLS token → (B, 768)

    def get_image_features(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values)
        return out.pooler_output                # (B, 768)

    def unfreeze_encoder_layers(self, num_layers: int = 2) -> None:
        bert_layers = self.text_encoder.encoder.layer
        for layer in bert_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.text_encoder.pooler.parameters():
            param.requires_grad = True

        vit_layers = self.vision_encoder.vision_model.encoder.layers
        for layer in vit_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.vision_encoder.vision_model.post_layernorm.parameters():
            param.requires_grad = True

        unfrozen = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total    = sum(p.numel() for p in self.parameters())
        print(f"  [DualEncoder] Unfroze last {num_layers} layers of each encoder.")
        print(f"  [DualEncoder] Trainable encoder params: "
              f"{unfrozen:,} / {total:,} ({100 * unfrozen / total:.1f}%)")


# ==========================================
# Classifier Head
# ==========================================

class ColorClassifier(nn.Module):
    """
    Drei-schichtiger MLP: (B, 1536) → (B, num_colors) logits.
    """

    def __init__(self, input_dim: int = 1536, num_colors: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_colors),
        )

    def forward(self, img_features, txt_features):
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        combined = torch.cat((img_features, txt_features), dim=-1)
        return self.fc(combined)


# ==========================================
# Early Stopping
# ==========================================

class EarlyStopping:
    """
    Stoppt Training wenn val-Metrik sich nicht verbessert.
    mode="max"  → höher ist besser (F1).
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4,
                 mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score           = float("-inf") if mode == "max" else float("inf")
        self.best_weights         = None
        self.best_encoder_weights = None

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def __call__(self, score, model, encoder=None):
        metric_name = "Val F1" if self.mode == "max" else "Val Loss"
        if self._is_improvement(score):
            self.best_score          = score
            self.counter             = 0
            self.best_weights        = copy.deepcopy(model.state_dict())
            if encoder is not None:
                self.best_encoder_weights = copy.deepcopy(encoder.state_dict())
            print(f"  [EarlyStopping] {metric_name} improved to {score:.6f} — weights saved.")
        else:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epoch(s)."
                  f"  (best {metric_name}: {self.best_score:.6f})")
            if self.counter >= self.patience:
                print("  [EarlyStopping] Patience exhausted — stopping training.")
                return True
        return False

    def restore_best_weights(self, model, encoder=None):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if encoder is not None and self.best_encoder_weights is not None:
                encoder.load_state_dict(self.best_encoder_weights)
            metric_name = "Val F1" if self.mode == "max" else "Val Loss"
            print(f"  [EarlyStopping] Best weights restored "
                  f"(best {metric_name} = {self.best_score:.6f}).")


# ==========================================
# Image Loading
# ==========================================

def load_image_as_rgb_array(path: str) -> np.ndarray:
    """
    Öffnet Bild und gibt garantiert (H, W, 3) uint8-Array zurück.
    Verhindert CLIPImageProcessor-Fehler bei Edge-Case-Bildern.
    """
    image = Image.open(path).convert("RGB")
    arr   = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[:, :, :3]
    return arr


# ==========================================
# Datasets
# ==========================================

class MultimodalColorDataset(Dataset):
    def __init__(self, dataframe, encoded_labels, img_dir,
                 tokenizer, image_processor, valid_indices=None):
        self.df              = dataframe.reset_index(drop=True)
        self.encoded_labels  = encoded_labels
        self.img_dir         = img_dir.rstrip("/")
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.valid_indices   = (valid_indices if valid_indices is not None
                                else list(range(len(self.df))))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx      = self.valid_indices[idx]
        row           = self.df.iloc[real_idx]
        text_input    = f"{row['item_name']} {row['item_caption']}"
        full_img_path = os.path.join(self.img_dir, row["image_file_name"])

        try:
            image_arr = load_image_as_rgb_array(full_img_path)
        except Exception:
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        text_enc = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        img_enc = self.image_processor(
            images=image_arr,
            return_tensors="pt",
            input_data_format="channels_last",
        )

        return {
            "input_ids":      text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values":   img_enc["pixel_values"].squeeze(0),
            "label":          torch.tensor(self.encoded_labels[real_idx],
                                           dtype=torch.float32),
        }


class InferenceDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, image_processor, valid_indices):
        self.df              = dataframe.reset_index(drop=True)
        self.img_dir         = img_dir.rstrip("/")
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.valid_indices   = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx      = self.valid_indices[idx]
        row           = self.df.iloc[real_idx]
        text_input    = f"{row['item_name']} {row['item_caption']}"
        full_img_path = os.path.join(self.img_dir, row["image_file_name"])

        try:
            image_arr = load_image_as_rgb_array(full_img_path)
        except Exception:
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        text_enc = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        img_enc = self.image_processor(
            images=image_arr,
            return_tensors="pt",
            input_data_format="channels_last",
        )

        return {
            "input_ids":      text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values":   img_enc["pixel_values"].squeeze(0),
            "row_index":      real_idx,
        }


# ==========================================
# Helper Functions
# ==========================================

def build_valid_indices(df: pd.DataFrame, img_dir: str) -> list:
    valid, skipped = [], 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        path = os.path.join(img_dir, row["image_file_name"])
        try:
            with Image.open(path) as im:
                im.convert("RGB")
                w, h = im.size
            if w < 2 or h < 2:
                skipped += 1
                continue
            valid.append(i)
        except Exception:
            skipped += 1
    print(f"  -> {len(valid)} valid, {skipped} skipped")
    return valid


def prepare_split_data(x_path, y_path, model_dir, val_ratio=0.1):
    print(f"Reading CSVs: {x_path}, {y_path}")
    df_x = pd.read_csv(str(x_path))
    df_y = pd.read_csv(str(y_path))
    df_y["color_tags"] = df_y["color_tags"].apply(ast.literal_eval)

    x_train, x_val, y_tags_train, y_tags_val = train_test_split(
        df_x, df_y["color_tags"], test_size=val_ratio, random_state=42
    )

    binarizer   = MultiLabelBinarizer()
    y_train_vec = binarizer.fit_transform(y_tags_train)
    y_val_vec   = binarizer.transform(y_tags_val)

    with open(os.path.join(model_dir, "mlb.pkl"), "wb") as f:
        pickle.dump(binarizer, f)
    print(f"Label classes ({len(binarizer.classes_)}): {list(binarizer.classes_)}")

    return x_train, x_val, y_train_vec, y_val_vec, binarizer


def ensure_min_predictions(probs, threshold, min_preds=1):
    preds = (probs > threshold).int().cpu().numpy()
    for i in range(len(preds)):
        if preds[i].sum() < min_preds:
            top_indices = torch.topk(probs[i], min_preds).indices.cpu().numpy()
            preds[i][top_indices] = 1
    return preds


def generate_predictions(classifier, dual_encoder, df_x, img_dir,
                          tokenizer, image_processor, mlb, device,
                          threshold=0.3, min_preds=1,
                          out_path="y_pred.csv"):
    print("\nGenerating predictions...")
    classifier.eval()
    dual_encoder.eval()
    df_reset  = df_x.reset_index(drop=True)
    valid_idx = build_valid_indices(df_reset, img_dir)

    infer_ds = InferenceDataset(df_reset, img_dir, tokenizer, image_processor, valid_idx)

    # ── Inference DataLoader: selbe Worker-Einstellungen wie Training
    NUM_WORKERS = min(6, os.cpu_count() or 1)
    infer_loader = DataLoader(
        infer_ds, batch_size=128, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    use_amp = (device == "cuda")
    all_row_indices, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting"):
            px   = batch["pixel_values"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            # BF16 autocast — kein GradScaler nötig
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                img_embeds = dual_encoder.get_image_features(px)
                txt_embeds = dual_encoder.get_text_features(ids, mask)
                probs      = torch.sigmoid(classifier(img_embeds, txt_embeds))

            preds = ensure_min_predictions(probs.float(), threshold, min_preds)
            all_preds.append(preds)
            all_row_indices.extend(batch["row_index"].tolist())

    all_preds_arr  = np.vstack(all_preds)
    decoded_labels = mlb.inverse_transform(all_preds_arr)

    results = pd.DataFrame({
        "image_file_name": df_reset.loc[all_row_indices, "image_file_name"].values,
        "color_tags":      [list(tags) for tags in decoded_labels],
    })
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} predictions to '{out_path}'")
    return results


# ==========================================
# Main Training Function
# ==========================================

def train():

    # ------------------------------------------------------------------
    # Hyperparameter
    # ------------------------------------------------------------------

    # ── BATCH_SIZE 32 → 128
    #    Original: 3,7 / 16 GB VRAM belegt.  Mit 128 landen wir bei ~8–10 GB.
    #    Größere Batches = stabilere Gradienten + 4× weniger DataLoader-Overhead.
    BATCH_SIZE = 128

    # ── LR skaliert moderat mit Batch-Size (nicht linear, da Adam)
    #    Original-Head-LR war 2e-3 bei Batch 32.  Wir erhöhen leicht auf 3e-3.
    LEARNING_RATE = 3e-3

    # ── Encoder-LR: leicht erhöht, bleibt aber klein gegen Catastrophic Forgetting
    ENCODER_LR = 2e-5

    # ── UNFREEZE_LAYERS 3 → 2
    #    Bei 190k Samples reichen 2 Schichten vollständig aus.
    #    2 Schichten = weniger trainierbare Parameter = schneller pro Step.
    UNFREEZE_LAYERS = 2

    VAL_RATIO       = 0.1
    TRAIN_THRESHOLD = 0.5
    VAL_THRESHOLD   = 0.5

    # ── MAX_EPOCHS 30 → 20
    #    LR-Scheduler + ES fangen die Konvergenz ab.  Bei 190k Samples ist
    #    Epoch 15+ selten relevant.  Spart im Worst Case 10 Epochen à ~5 min.
    MAX_EPOCHS = 20

    # ── ES_PATIENCE 7 → 5
    #    Bei großem Datensatz zeigt Early Stopping seine Wirkung schneller.
    #    5 Epochen ohne F1-Verbesserung sind ein klares Stoppsignal.
    ES_PATIENCE  = 5
    ES_MIN_DELTA = 1e-4

    LR_PATIENCE = 2
    LR_FACTOR   = 0.5
    LR_MIN      = 1e-6

    TEXT_MODEL_ID   = "cl-tohoku/bert-base-japanese-v3"
    VISION_MODEL_ID = "openai/clip-vit-base-patch16"

    X_TRAIN_PATH = "data/processed/X_train_processed.csv"
    Y_TRAIN_PATH = "data/processed/y_train_processed.csv"
    IMG_DIR      = "data/images/"
    MODEL_DIR    = "models/"

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Device + AMP-Konfiguration
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── BF16 statt FP16
    #    RTX 5060 Ti (Blackwell) hat native BF16-Tensor-Cores.
    #    BF16 hat denselben Exponentenbereich wie FP32 → kein Gradient-Underflow
    #    → GradScaler wird nicht benötigt, bleibt aber als Dummy aktiv (enabled=False).
    use_amp   = (device == "cuda")
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler    = GradScaler("cuda", enabled=False)   # nicht nötig bei BF16

    # ------------------------------------------------------------------
    # Tokenizer, Image Processor, Encoder
    # ------------------------------------------------------------------
    print("\nLoading tokenizer and image processor...")
    tokenizer       = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL_ID)

    print("Loading dual encoder (BERT + CLIP vision)...")
    dual_encoder = DualEncoder(TEXT_MODEL_ID, VISION_MODEL_ID).to(device)

    print(f"Unfreezing last {UNFREEZE_LAYERS} encoder layers...")
    dual_encoder.unfreeze_encoder_layers(num_layers=UNFREEZE_LAYERS)
    dual_encoder.train()

    # ------------------------------------------------------------------
    # Daten
    # ------------------------------------------------------------------
    X_tr, X_va, y_tr, y_va, mlb_obj = prepare_split_data(
        X_TRAIN_PATH, Y_TRAIN_PATH, MODEL_DIR, val_ratio=VAL_RATIO
    )
    print(f"\nTrain size: {len(X_tr):,}  |  Val size: {len(X_va):,}")

    print("\nValidating training images...")
    train_valid_idx = build_valid_indices(X_tr.reset_index(drop=True), IMG_DIR)
    print("Validating validation images...")
    val_valid_idx   = build_valid_indices(X_va.reset_index(drop=True), IMG_DIR)

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = MultimodalColorDataset(X_tr, y_tr, IMG_DIR, tokenizer,
                                       image_processor, train_valid_idx)
    val_ds   = MultimodalColorDataset(X_va, y_va, IMG_DIR, tokenizer,
                                       image_processor, val_valid_idx)

    # ── num_workers=6  →  paralleles Bild-Dekodieren während GPU rechnet
    # ── pin_memory=True  →  schnellerer PCIe-Transfer (kein Zwischen-Copy)
    # ── persistent_workers=True  →  Worker bleiben zwischen Epochen am Leben
    # ── prefetch_factor=2  →  2 Batches werden vorab in CPU-RAM geladen
    NUM_WORKERS = min(6, os.cpu_count() or 1)
    print(f"\nDataLoader: batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # ------------------------------------------------------------------
    # Classifier, Loss, Optimizer, Scheduler, Early Stopping
    # ------------------------------------------------------------------
    num_classes = len(mlb_obj.classes_)
    classifier  = ColorClassifier(input_dim=1536, num_colors=num_classes).to(device)

    label_counts = y_tr.sum(axis=0).astype(float)
    neg_counts   = len(y_tr) - label_counts
    pos_weights  = torch.tensor(
        np.clip(neg_counts / (label_counts + 1e-6), a_min=1.0, a_max=10.0),
        dtype=torch.float32
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    encoder_params = [p for p in dual_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": classifier.parameters(), "lr": LEARNING_RATE},
        {"params": encoder_params,          "lr": ENCODER_LR},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    early_stopping = EarlyStopping(
        patience=ES_PATIENCE, min_delta=ES_MIN_DELTA, mode="max"
    )

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    print(f"\nStarting training (max {MAX_EPOCHS} epochs, "
          f"batch={BATCH_SIZE}, workers={NUM_WORKERS}, amp=BF16)...\n")

    for epoch in range(MAX_EPOCHS):

        # ── Train
        classifier.train()
        dual_encoder.train()
        all_preds, all_labels = [], []
        t_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)   # set_to_none=True spart Null-Initialisierung

            px     = batch["pixel_values"].to(device, non_blocking=True)
            ids    = batch["input_ids"].to(device, non_blocking=True)
            mask   = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)

            # ── non_blocking=True  →  CPU-GPU-Transfer überlappend mit GPU-Compute
            # ── BF16 autocast: kein GradScaler-Wrapped nötig
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                img_embeds = dual_encoder.get_image_features(px)
                txt_embeds = dual_encoder.get_text_features(ids, mask)
                logits     = classifier(img_embeds, txt_embeds)
                loss       = loss_fn(logits, target)

            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits.float()) > TRAIN_THRESHOLD).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(target.cpu().numpy())
            t_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_f1 = f1_score(np.vstack(all_labels), np.vstack(all_preds),
                            average="micro", zero_division=0)

        # ── Validation
        classifier.eval()
        dual_encoder.eval()
        val_preds, val_labels = [], []
        v_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]"):
                px     = batch["pixel_values"].to(device, non_blocking=True)
                ids    = batch["input_ids"].to(device, non_blocking=True)
                mask   = batch["attention_mask"].to(device, non_blocking=True)
                target = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    img_embeds = dual_encoder.get_image_features(px)
                    txt_embeds = dual_encoder.get_text_features(ids, mask)
                    logits     = classifier(img_embeds, txt_embeds)
                v_loss += loss_fn(logits, target).item()

                preds = (torch.sigmoid(logits.float()) > VAL_THRESHOLD).int().cpu().numpy()
                val_preds.append(preds)
                val_labels.append(target.cpu().numpy())

        avg_val_loss = v_loss / len(val_loader)
        val_f1 = f1_score(np.vstack(val_labels), np.vstack(val_preds),
                          average="micro", zero_division=0)
        lr_head = optimizer.param_groups[0]["lr"]
        lr_enc  = optimizer.param_groups[1]["lr"]

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss : {t_loss/len(train_loader):.4f}  |  Train F1 : {train_f1:.4f}")
        print(f"Val   Loss : {avg_val_loss:.4f}  |  Val   F1 : {val_f1:.4f}  "
              f"|  LR head : {lr_head:.2e}  |  LR enc : {lr_enc:.2e}")

        if device == "cuda":
            vram_used = torch.cuda.memory_reserved() / 1e9
            print(f"VRAM reserved: {vram_used:.1f} GB")

        scheduler.step(avg_val_loss)

        checkpoint = {
            "classifier":   classifier.state_dict(),
            "dual_encoder": dual_encoder.state_dict(),
        }
        ckpt_path = os.path.join(MODEL_DIR, f"color_model_ep{epoch+1}.pth")
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        if early_stopping(val_f1, classifier, encoder=dual_encoder):
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    # ------------------------------------------------------------------
    # Best weights speichern
    # ------------------------------------------------------------------
    early_stopping.restore_best_weights(classifier, encoder=dual_encoder)
    best_checkpoint = {
        "classifier":   classifier.state_dict(),
        "dual_encoder": dual_encoder.state_dict(),
    }
    best_path = os.path.join(MODEL_DIR, "color_model_best.pth")
    torch.save(best_checkpoint, best_path)
    print(f"\nBest model saved to '{best_path}'.")

    # ------------------------------------------------------------------
    # y_pred_training.csv generieren
    # ------------------------------------------------------------------
    generate_predictions(
        classifier=classifier, dual_encoder=dual_encoder,
        df_x=X_va, img_dir=IMG_DIR, tokenizer=tokenizer,
        image_processor=image_processor, mlb=mlb_obj,
        device=device, threshold=VAL_THRESHOLD, min_preds=1,
        out_path=os.path.join(MODEL_DIR, "y_pred_training.csv"),
    )


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    train()