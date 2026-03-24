"""
XLM-RoBERTa Fine-Tuning fuer Multi-Label Color Classification.

Nutzung:
    python -m src.models.train_model                # Training mit Default-Config
    python -m src.models.train_model --epochs 5     # Config ueberschreiben
"""
import torch
import torch.nn as nn
import numpy as np
import ast
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS, NUM_LABELS, XLM_CONFIG, MODEL_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════
class RakutenTextDataset(Dataset):
    """Dataset for Color Classification, Text Based."""

    def __init__(self, df_x, df_y, tokenizer, max_len=256):
        self.df_x      = df_x.reset_index(drop=True)
        self.df_y      = df_y.reset_index(drop=True) if df_y is not None else None
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, idx):
        row  = self.df_x.iloc[idx]
        text = f"{row['item_name']} [SEP] {row.get('item_caption', '')}"
        text = text[:512]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = torch.zeros(NUM_LABELS)
        if self.df_y is not None:
            raw  = self.df_y.iloc[idx]['color_tags']
            tags = ast.literal_eval(raw) if isinstance(raw, str) else raw        #correction of "["green, "blue"]"m the outer ""
            for tag in tags:
                if tag in COLOR_LABELS:
                    labels[COLOR_LABELS.index(tag)] = 1.0

        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         labels,
        }


# ═══════════════════════════════════════════════════════════════
# Modell
# ═══════════════════════════════════════════════════════════════
class TextColorClassifier(nn.Module):
    """XLM-RoBERTa + Linear Head for multi-Label Classification, Training"""

    def __init__(self, model_name, num_labels=NUM_LABELS, dropout=0.3):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(self.dropout(pooled))


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════
def train_xlm(train_x, train_y, val_x, val_y, config=None):
    """
    trains XLM-RoBERTa and return the model + Thresholds

    Returns:
        model, thresholds, run_id
    """
    config = {**XLM_CONFIG, **(config or {})}

    # Optional: MLflow Tracking
    run_id = None
    mlflow_active = False
    try:
        import mlflow
        import mlflow.pytorch
        mlflow.set_experiment("rakuten_color_extraction")
        run_ctx = mlflow.start_run(run_name=f"xlm_roberta_lr{config['lr']}")
        run_ctx.__enter__()
        mlflow.log_params(config)
        run_id = mlflow.active_run().info.run_id
        mlflow_active = True
        print(f"MLflow Run: {run_id}")
    except Exception as e:
        print(f"MLflow nicht verfuegbar ({e}), trainiere ohne Tracking...")
        run_id = f"local_{np.random.randint(10000, 99999)}"

    print(f"\n{'='*60}")
    print(f"XLM-RoBERTa Training")
    print(f"  Device:     {DEVICE}")
    print(f"  LR:         {config['lr']}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Len:    {config['max_len']}")
    print(f"{'='*60}\n")

    # ── Tokenizer + Datasets ──
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    train_ds  = RakutenTextDataset(train_x, train_y, tokenizer, config['max_len'])
    val_ds    = RakutenTextDataset(val_x,   val_y,   tokenizer, config['max_len'])

    num_workers = 0 if DEVICE == "cpu" else 4
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'],
                          shuffle=True,  num_workers=num_workers, pin_memory=(DEVICE=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=config['batch_size']*2,
                          shuffle=False, num_workers=num_workers, pin_memory=(DEVICE=="cuda"))

    # ── Modell + Loss ──
    model = TextColorClassifier(
        config['model_name'],
        num_labels=NUM_LABELS,
        dropout=config['dropout']
    ).to(DEVICE)

    pos_weight = torch.ones(NUM_LABELS) * config.get('pos_weight_factor', 1.0)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    total_steps = len(train_dl) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.get('warmup_ratio', 0.1)),
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    best_model_path = MODEL_DIR / "xlm_best.pt"

    # ── Epoch Loop ──
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config['epochs']} Train"):
            ids   = batch['input_ids'].to(DEVICE)
            mask  = batch['attention_mask'].to(DEVICE)
            lbls  = batch['labels'].to(DEVICE)

            logits = model(ids, mask)
            loss   = criterion(logits, lbls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dl)

        # Validation
        val_f1, val_probs = evaluate(model, val_dl)

        print(f"  Epoch {epoch+1}: loss={avg_train_loss:.4f}  val_wF1={val_f1:.4f}")

        if mlflow_active:
            import mlflow
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_weighted_f1": val_f1,
            }, step=epoch)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best-Modell saved (F1={val_f1:.4f})")

            if mlflow_active:
                import mlflow.pytorch
                mlflow.pytorch.log_model(model, "model_best")

    # ── Thresholds optimization ── optional
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    _, val_probs = evaluate(model, val_dl)
    thresholds = optimize_thresholds(val_probs, val_y)

    if mlflow_active:
        import mlflow
        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.active_run()  # keep alive for pipeline
        run_ctx.__exit__(None, None, None)

    # Run in DB speichern
    try:
        from src.db import save_run
        save_run(run_id, "xlm_roberta", best_val_f1, config)
    except Exception:
        pass

    print(f"\nTraining ready! Best Val F1: {best_val_f1:.4f}")
    return model, thresholds, run_id


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════
def evaluate(model, dataloader):
    """Returns weighted-F1 and probability-Matrix"""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            logits = model(ids, mask)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch['labels'].numpy())

    prob_matrix  = np.vstack(all_probs)
    label_matrix = np.vstack(all_labels)

    preds = (prob_matrix >= 0.5).astype(int)
    weighted_f1 = f1_score(label_matrix, preds, average='weighted', zero_division=0)

    return weighted_f1, prob_matrix


def get_text_probs(model, df_x, tokenizer, max_len=256, batch_size=64):
    """Returns probability-scores for df (e.x. Test-Set)."""
    ds = RakutenTextDataset(df_x, df_y=None, tokenizer=tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dl:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            probs = torch.sigmoid(model(ids, mask)).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


# ═══════════════════════════════════════════════════════════════
# Threshold Optimization
# ═══════════════════════════════════════════════════════════════
def optimize_thresholds(prob_matrix, df_y):
    """Findet optimalen Threshold pro Klasse via Grid Search."""
    y_true = np.zeros((len(prob_matrix), NUM_LABELS))
    for i, (_, row) in enumerate(df_y.reset_index(drop=True).iterrows()):
        raw  = row['color_tags']
        tags = ast.literal_eval(raw) if isinstance(raw, str) else raw
        for tag in tags:
            if tag in COLOR_LABELS:
                y_true[i, COLOR_LABELS.index(tag)] = 1

    thresholds = {}
    for j, color in enumerate(COLOR_LABELS):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (prob_matrix[:, j] >= t).astype(int)
            f1 = f1_score(y_true[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        thresholds[color] = round(best_t, 2)

    # Gesamt weighted-F1
    t_arr = np.array([thresholds[c] for c in COLOR_LABELS])
    preds = (prob_matrix >= t_arr).astype(int)
    wf1 = f1_score(y_true, preds, average='weighted', zero_division=0)
    print(f"  Weighted F1 nach Threshold-Opt: {wf1:.4f}")
    print(f"  Thresholds: { {k: v for k, v in thresholds.items() if v != 0.5} }")

    return thresholds


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.config import DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=XLM_CONFIG['epochs'])
    parser.add_argument("--lr", type=float, default=XLM_CONFIG['lr'])
    parser.add_argument("--batch_size", type=int, default=XLM_CONFIG['batch_size'])
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()

    # Daten laden
    if args.real:
        df_x = pd.read_csv(DATA_DIR / "X_train_12tkObq.csv")
        df_y = pd.read_csv(DATA_DIR / "y_train_Q9n2dCu.csv")
    else:
        df_x = pd.read_csv(DATA_DIR / "X_train.csv")
        df_y = pd.read_csv(DATA_DIR / "y_train.csv")

    train_x, val_x, train_y, val_y = train_test_split(
        df_x, df_y, test_size=0.2, random_state=42
    )

    config = {**XLM_CONFIG, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}
    model, thresholds, run_id = train_xlm(train_x, train_y, val_x, val_y, config)