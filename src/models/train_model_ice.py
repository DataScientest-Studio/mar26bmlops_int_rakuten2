"""
================================================================================
Rakuten Color Classifier
================================================================================

TASK
----
Given a Japanese e-commerce item (image + Japanese title + Japanese caption),
predict one or more colour tags (e.g. "red", "navy", "white") from a fixed
label set.  This is a multi-label classification problem: a single item
can belong to several colour classes.

ARCHITECTURE OVERVIEW
---------------------
We use a "partially fine-tuned dual-encoder + trainable MLP head" approach:

  1. Text encoder  : cl-tohoku/bert-base-japanese-v3

  2. Image encoder : openai/clip-vit-base-patch16

  3. Classifier head : a three-layer MLP

OUTPUT
------
  - color_model_best.pth  : weights of the epoch with the best val F1
  - color_model_epN.pth   : per-epoch checkpoints in the same dict format
  - mlb.pkl               : fitted MultiLabelBinarizer
  - y_pred_training.csv   : predictions on the validation split
                            columns: image_file_name | color_tags

USAGE
-----
    Reads  : data/processed/X_train_processed.csv
             data/processed/y_train_processed.csv
             data/images/
    Writes : models/color_model_best.pth
             models/color_model_epN.pth  (one per epoch)
             models/mlb.pkl
             models/y_pred_training.csv
================================================================================
"""
from torch.amp import autocast # mk                         24.03.2026n new line
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
from torch.amp import autocast, GradScaler           #mk instead of:  from torch.cuda.amp import autocast as cuda_autocast, GradScaler
from contextlib import contextmanager

#@contextmanager                                    # mk produces error
# def autocast(device_type, enabled=True):
#     with autocast("cuda", enabled=enabled):
#         yield
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, CLIPImageProcessor

# Allow PIL to load truncated image files rather than raising OSError.
# im.verify() only checks the file header — a file can pass verification
# but still be truncated (missing pixel data at the end).  This setting
# tells PIL to decode as much data as it can and fill the rest with grey
# rather than crashing.  Applied once at module load time.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========================================
# Dual Encoder
# ==========================================

class DualEncoder(nn.Module):
    """
    Wraps a Japanese BERT text encoder and a CLIP vision encoder.

    All parameters are frozen on initialisation.  After instantiation,
    unfreeze_encoder_layers() is called in train() to selectively unfreeze
    the last N transformer layers of each encoder for fine-tuning.

    Args:
        text_model_id   : HuggingFace model ID for the Japanese BERT model.
        vision_model_id : HuggingFace model ID for the CLIP vision model.

    Forward interface:
        get_text_features(input_ids, attention_mask) -> Tensor (B, 768)
        get_image_features(pixel_values)             -> Tensor (B, 768)
    """

    def __init__(self, text_model_id: str, vision_model_id: str):
        super().__init__()

        # Japanese BERT — full 12-layer transformer with a WordPiece tokeniser
        # trained on Japanese text
        self.text_encoder = AutoModel.from_pretrained(text_model_id)

        # CLIP ViT-B/16 vision tower
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_id)

        # Freeze all parameters in both encoders on initialisation.
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def get_text_features(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tokenised Japanese strings.

        Returns:
            Tensor of shape (batch_size, 768).
        """
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, seq_len, 768)  →  [:, 0, :] picks CLS token
        return out.last_hidden_state[:, 0, :]

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of preprocessed images.

        Returns:
            Tensor of shape (batch_size, 768).
        """
        out = self.vision_encoder(pixel_values=pixel_values)
        return out.pooler_output

    def unfreeze_encoder_layers(self, num_layers: int = 2) -> None:
        """
        Unfreeze the last `num_layers` transformer layers of each encoder.

        Args:
            num_layers : number of final transformer layers to unfreeze in
                         each encoder (default 2).  Start with 2; increase
                         to 3–4 if val F1 is still plateauing.
        """
        # BERT: unfreeze the last `num_layers` encoder layers ---
        bert_layers = self.text_encoder.encoder.layer
        for layer in bert_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze the final LayerNorm that follows the transformer stack,
        # since it directly transforms the output fed to our classifier head.
        for param in self.text_encoder.pooler.parameters():
            param.requires_grad = True

        # CLIP ViT: unfreeze the last `num_layers` transformer layers ---
        vit_layers = self.vision_encoder.vision_model.encoder.layers
        for layer in vit_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze the post-layernorm applied after the transformer stack
        for param in self.vision_encoder.vision_model.post_layernorm.parameters():
            param.requires_grad = True

        # Count and report unfrozen parameters so the team can see what changed
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
    Three-layer MLP that maps a concatenated image and text embedding to
    per-class logits for multi-label colour prediction.

    Input  : concatenation of L2-normalised image and text embeddings
             → shape (batch_size, input_dim)  where input_dim = 768 + 768 = 1536
    Output : raw logits (batch_size, num_colors)
             Apply sigmoid + threshold at inference to get binary predictions.

    Args:
        input_dim  : dimensionality of the concatenated embedding (default 1536).
        num_colors : number of colour classes to predict.
    """

    def __init__(self, input_dim: int = 1536, num_colors: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),  # wider first layer for larger dataset
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_colors),
        )

    def forward(self, img_features: torch.Tensor,
                txt_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_features : raw image embeddings  (B, 768)
            txt_features : raw text  embeddings  (B, 768)

        Returns:
            logits : (B, num_colors) 
        """
        # L2-normalise both embeddings so that the dot product between any
        # two feature vectors equals their cosine similarity (range [-1, 1]).
        # This prevents one modality from dominating simply due to scale.
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        # Concatenate along the feature dimension → (B, 1536)
        combined = torch.cat((img_features, txt_features), dim=-1)
        return self.fc(combined)


# ==========================================
# Early Stopping Callback
# ==========================================

class EarlyStopping:
    """
    Monitors a validation metric and stops training when it stops improving.

    Supports both "higher is better" (e.g. F1) and "lower is better" (e.g. loss)
    via the `mode` argument.  Defaults to "max" (F1) since F1 more directly
    reflects what we care about — correct colour predictions — while val loss
    can plateau even as F1 continues to improve.

    Note: ReduceLROnPlateau still monitors val loss (passed separately in the
    training loop) because the scheduler is designed to escape loss plateaus
    by reducing the LR — which is a different concern from when to stop entirely.

    Args:
        patience  : epochs to wait without improvement before stopping.
        min_delta : minimum change that counts as a genuine improvement.
                    For F1 (range 0–1) a delta of 1e-4 means 0.01% improvement.
        mode      : "max" for metrics where higher is better (F1, accuracy),
                    "min" for metrics where lower is better (loss, error rate).
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        # best starts at -inf for "max" mode, +inf for "min" mode
        self.best_score   = float("-inf") if mode == "max" else float("inf")
        self.best_weights         = None
        self.best_encoder_weights = None

    def _is_improvement(self, score: float) -> bool:
        """Return True if `score` is a meaningful improvement over best so far."""
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def __call__(self, score: float, model: nn.Module,
                 encoder: nn.Module = None) -> bool:
        """
        Call once per epoch.

        Args:
            score   : current epoch metric value (val F1 by default).
            model   : classifier head — weights saved on improvement.
            encoder : dual encoder — weights also saved since last layers
                      are fine-tuned and must stay in sync with the head.

        Returns:
            True  if patience is exhausted and training should stop.
            False if training should continue.
        """
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

    def restore_best_weights(self, model: nn.Module,
                             encoder: nn.Module = None) -> None:
        """
        Restore the best-seen weights into the model and encoder.
        Always call this after the training loop.
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if encoder is not None and self.best_encoder_weights is not None:
                encoder.load_state_dict(self.best_encoder_weights)
            metric_name = "Val F1" if self.mode == "max" else "Val Loss"
            print(f"  [EarlyStopping] Best weights restored "
                  f"(best {metric_name} = {self.best_score:.6f}).")


# ==========================================
# Datasets
# ==========================================

def load_image_as_rgb_array(path: str) -> np.ndarray:
    """
    Open an image file and return it as a guaranteed (H, W, 3) uint8 numpy
    array suitable for CLIPImageProcessor.

    Why this is needed:
      CLIPImageProcessor infers the channel dimension from the array shape.
      On unusual images — tiny 1×1 thumbnails, paletted PNGs, images that
      survive PIL's convert("RGB") in a non-standard internal format — the
      processor can misidentify the channel axis and raise:

          ValueError: mean must have 1 elements if it is an iterable, got 3

      Passing an explicit (H, W, 3) numpy array removes all ambiguity.
      This issue only surfaces at scale (50k+ images) because the dataset
      contains a wider variety of edge-case files.

    Handles:
      - Grayscale images that remain 2-D after convert("RGB")
      - RGBA / 4-channel images — alpha channel is dropped
      - Any other unexpected channel count — truncated/repeated to 3

    Args:
        path : absolute path to the image file.

    Returns:
        numpy array of shape (H, W, 3), dtype uint8.
    """
    image = Image.open(path).convert("RGB")
    arr   = np.array(image)

    if arr.ndim == 2:
        # Still grayscale — stack into 3 identical channels
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] != 3:
        # e.g. RGBA (4 channels) — keep only the first 3
        arr = arr[:, :, :3]

    return arr


class MultimodalColorDataset(Dataset):
    """
    PyTorch Dataset for training and validation.

    Each item yields tokenised text, preprocessed image pixels, and a
    multi-hot label vector.

    Args:
        dataframe      : X DataFrame with columns:
                           item_name       — Japanese product title
                           item_caption    — Japanese product description
                           image_file_name — filename relative to img_dir
        encoded_labels : numpy array (N, num_classes) produced by
                         MultiLabelBinarizer.  Row i is the multi-hot vector
                         for dataframe row i.
        img_dir        : root directory containing image files.
        tokenizer      : Japanese BERT tokenizer (AutoTokenizer).
        image_processor: CLIP image preprocessor (CLIPImageProcessor).
        valid_indices  : list of integer row indices that passed image
                         validation.  Only these rows are served by __getitem__.
                         Pre-computed by build_valid_indices() to avoid any
                         runtime I/O errors inside the training loop.
    """

    def __init__(self, dataframe, encoded_labels, img_dir,
                 tokenizer, image_processor, valid_indices=None):
        # reset_index ensures iloc indexing is contiguous 
        self.df              = dataframe.reset_index(drop=True)
        self.encoded_labels  = encoded_labels
        self.img_dir         = img_dir.rstrip("/")
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.valid_indices   = (valid_indices if valid_indices is not None
                                else list(range(len(self.df))))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        # Map the dataset-local index back to the original DataFrame row.
        real_idx      = self.valid_indices[idx]
        row           = self.df.iloc[real_idx]

        # Concatenate title + caption into a single Japanese string.
        # The BERT tokenizer handles the combined string as one sequence.
        text_input    = f"{row['item_name']} {row['item_caption']}"
        full_img_path = os.path.join(self.img_dir, row["image_file_name"])

        try:
            # load_image_as_rgb_array guarantees a (H, W, 3) uint8 array,
            # preventing CLIPImageProcessor from misidentifying the channel axis
            # on unusual images (tiny thumbnails, paletted PNGs, RGBA, etc.).
            # LOAD_TRUNCATED_IMAGES=True means PIL fills missing pixel data with
            # grey rather than raising OSError on truncated files.
            image_arr = load_image_as_rgb_array(full_img_path)
        except Exception:
            # Last-resort fallback: if the image still fails after all the
            # above guards, substitute a plain grey 224×224 image so the
            # batch can proceed without crashing.  This is rare — build_valid_indices
            # now does a full pixel decode, so only files that become corrupt
            # between validation and training will reach here.
            image_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        # Tokenise the Japanese text.
        # padding="max_length" + truncation=True ensures every batch has the
        # same sequence length (128 tokens), which is required for DataLoader
        # to stack tensors into a batch without custom collation.
        text_enc = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        # Preprocess the image: resize to 224×224, normalise with CLIP's
        # ImageNet-derived mean and std.
        # input_data_format="channels_last" explicitly tells the processor
        # that the last axis is the channel dimension (H, W, C).  Without
        # this, 1×1 pixel images (shape (1, 1, 3)) are ambiguous — the
        # processor may guess C is the first axis and fail with:
        #   ValueError: mean must have 1 elements if it is an iterable, got 3
        img_enc = self.image_processor(
            images=image_arr,
            return_tensors="pt",
            input_data_format="channels_last",
        )

        return {
            # squeeze(0) removes the batch dimension added by return_tensors="pt"
            # so that DataLoader can re-batch correctly.
            "input_ids":      text_enc["input_ids"].squeeze(0),       # (128,)
            "attention_mask": text_enc["attention_mask"].squeeze(0),  # (128,)
            "pixel_values":   img_enc["pixel_values"].squeeze(0),     # (3, 224, 224)
            "label":          torch.tensor(
                                  self.encoded_labels[real_idx],
                                  dtype=torch.float32                  # (num_classes,)
                              ),
        }


class InferenceDataset(Dataset):
    """
    Inference-only variant of MultimodalColorDataset.

    Identical to the training dataset except:
      - No labels are loaded (we don't have them at inference time).
      - Returns `row_index` so that generate_predictions() can align each
        prediction back to the correct row in the original DataFrame when
        writing y_pred.csv.

    Args: same as MultimodalColorDataset except no encoded_labels.
    """

    def __init__(self, dataframe, img_dir, tokenizer, image_processor, valid_indices):
        self.df              = dataframe.reset_index(drop=True)
        self.img_dir         = img_dir.rstrip("/")
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.valid_indices   = valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
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
            "row_index":      real_idx,  # used to align predictions with df rows
        }


# ==========================================
# Helper Functions
# ==========================================

def build_valid_indices(df: pd.DataFrame, img_dir: str) -> list:
    """
    Pre-validate all image files in a DataFrame before training begins.

    Iterates over every row and attempts to open + verify the image file.
    Only indices of successfully opened files are returned.

    Args:
        df      : DataFrame containing an "image_file_name" column.
        img_dir : Root directory for image files.

    Returns:
        List of integer row indices (into df) that are safe to use.
    """
    valid, skipped = [], 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        path = os.path.join(img_dir, row["image_file_name"])
        try:
            with Image.open(path) as im:
                # Full decode (not just header check) catches truncated files
                # that pass im.verify() but crash during training when PIL
                # tries to read missing pixel data.  Converting to RGB forces
                # a complete decode of all pixel data.
                im.convert("RGB")
                w, h = im.size

            # Skip 1×1 pixel images — they carry no visual information and
            # cause CLIPImageProcessor to mis-detect the channel axis.
            if w < 2 or h < 2:
                skipped += 1
                continue

            valid.append(i)
        except Exception:
            skipped += 1
    print(f"  -> {len(valid)} valid, {skipped} skipped")
    return valid


def prepare_split_data(x_path: str, y_path: str, model_dir: str, val_ratio: float = 0.1):
    """
    Load, split, and binarize the training data.

    Steps:
      1. Load X (features) and y (colour tags) CSVs.
      2. Parse colour tag strings into Python lists with ast.literal_eval.
      3. Stratified 90/10 train/val split (random_state=42 for reproducibility).
      4. Fit a MultiLabelBinarizer on the training split and transform both
         splits into multi-hot numpy arrays.
      5. Save the fitted binarizer to mlb.pkl so it can be reloaded at
         inference time without re-fitting.

    Args:
        x_path    : path to X_train.csv
        y_path    : path to y_train.csv
        model_dir : directory where mlb.pkl will be saved
        val_ratio : fraction of data to hold out for validation (default 10%)

    Returns:
        x_train, x_val          : DataFrames
        y_train_vec, y_val_vec  : numpy arrays (N, num_classes), dtype int
        binarizer               : fitted MultiLabelBinarizer
    """
    print(f"Reading CSVs: {x_path}, {y_path}")
    df_x = pd.read_csv(str(x_path))
    df_y = pd.read_csv(str(y_path))

    # colour_tags are stored as string representations of Python lists,
    # e.g. "['red', 'white']".  literal_eval safely converts them back.
    df_y["color_tags"] = df_y["color_tags"].apply(ast.literal_eval)

    x_train, x_val, y_tags_train, y_tags_val = train_test_split(
        df_x, df_y["color_tags"],
        test_size=val_ratio,
        random_state=42,   # fixed seed → same split every run
    )

    # MultiLabelBinarizer converts a list of lists of labels into a 2-D
    # binary matrix.  E.g. [['red', 'white'], ['blue']] → [[1,0,1],[0,1,0]]
    binarizer   = MultiLabelBinarizer()
    y_train_vec = binarizer.fit_transform(y_tags_train)  # fit + transform
    y_val_vec   = binarizer.transform(y_tags_val)         # transform only

    # Persist the binarizer — it encodes the class vocabulary and ordering,
    # which must match exactly when decoding predictions at inference time.
    with open(os.path.join(model_dir, "mlb.pkl"), "wb") as f:
        pickle.dump(binarizer, f)
    print(f"Label classes ({len(binarizer.classes_)}): {list(binarizer.classes_)}")

    return x_train, x_val, y_train_vec, y_val_vec, binarizer


# ==========================================
# 5b. Minimum Prediction Helper
# ==========================================

def ensure_min_predictions(probs: torch.Tensor, threshold: float,
                            min_preds: int = 1) -> np.ndarray:
    """
    Apply the confidence threshold as normal, but guarantee that every item
    receives at least `min_preds` colour predictions.

    For any row where no class clears the threshold, the top-N highest
    probability classes are forced to 1 regardless of their confidence score.
    This prevents the model from returning empty lists on uncertain items —
    it will always commit to its best guess.

    Args:
        probs     : sigmoid probabilities, shape (B, num_classes).
                    Must be on CPU before calling this function.
        threshold : normal confidence cutoff (e.g. 0.3).
        min_preds : minimum number of colours to predict per item (default 1).

    Returns:
        Binary numpy array of shape (B, num_classes).
    """
    preds = (probs > threshold).int().cpu().numpy()

    for i in range(len(preds)):
        if preds[i].sum() < min_preds:
            # Fewer than min_preds classes cleared the threshold for this item.
            # Force the top-N classes by probability so we always output
            # at least min_preds colours.
            top_indices       = torch.topk(probs[i], min_preds).indices.cpu().numpy()
            preds[i][top_indices] = 1

    return preds


# ==========================================
# 6. Prediction → y_pred.csv
# ==========================================

def generate_predictions(classifier: nn.Module, dual_encoder: nn.Module,
                          df_x: pd.DataFrame, img_dir: str,
                          tokenizer, image_processor,
                          mlb: MultiLabelBinarizer, device: str,
                          threshold: float = 0.3,
                          min_preds: int = 1,
                          out_path: str = "y_pred.csv") -> pd.DataFrame:
    """
    Run inference over a DataFrame and write predictions to a CSV file.

    Output CSV columns:
        image_file_name : original filename from df_x
        color_tags      : Python list of predicted colour strings, e.g.
                          "['red', 'white']"

    Args:
        classifier      : trained ColorClassifier (best weights already loaded).
        dual_encoder    : frozen DualEncoder.
        df_x            : DataFrame to predict on (validation or test split).
        img_dir         : root image directory.
        tokenizer       : Japanese BERT tokenizer.
        image_processor : CLIP image preprocessor.
        mlb             : fitted MultiLabelBinarizer.
        device          : "cuda" or "cpu".
        threshold       : sigmoid probability cutoff for a positive prediction.
        min_preds       : minimum number of colours to always predict per item.
        out_path        : path to write the output CSV.

    Returns:
        DataFrame with columns [image_file_name, color_tags].
    """
    print("\nGenerating predictions...")
    classifier.eval()
    dual_encoder.eval()
    df_reset  = df_x.reset_index(drop=True)
    valid_idx = build_valid_indices(df_reset, img_dir)

    infer_ds = InferenceDataset(
        df_reset, img_dir, tokenizer, image_processor, valid_idx
    )

    # num_workers=0, pin_memory=False — same reason as training loaders.
    infer_loader = DataLoader(
        infer_ds, batch_size=32, shuffle=False,
        num_workers=0, pin_memory=False
    )

    use_amp = (device == "cuda")

    all_row_indices: list = []
    all_preds: list       = []

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting"):
            px   = batch["pixel_values"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with autocast("cuda", enabled=use_amp):
                img_embeds = dual_encoder.get_image_features(px)
                txt_embeds = dual_encoder.get_text_features(ids, mask)
                probs      = torch.sigmoid(classifier(img_embeds, txt_embeds))

            # Apply threshold, guaranteeing at least min_preds colours per
            # item so the model never returns an empty list.
            preds = ensure_min_predictions(probs, threshold, min_preds)
            all_preds.append(preds)
            all_row_indices.extend(batch["row_index"].tolist())

    # Stack list of (batch_size, num_classes) arrays → (N, num_classes)
    all_preds_arr  = np.vstack(all_preds)

    # Decode multi-hot rows back to tuples of colour strings
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
    """
    Full training pipeline.

    Execution order:
      1. Load models and processors.
      2. Load and split data; binarize labels.
      3. Pre-validate images; build Datasets and DataLoaders.
      4. Initialise classifier head, loss function, optimiser, scheduler,
         and early stopping callback.
      5. Training loop (up to MAX_EPOCHS, subject to early stopping):
           a. Forward pass through frozen encoders + trainable head.
           b. Compute BCEWithLogitsLoss; backward; optimiser step.
           c. Validation pass; compute val loss and micro-F1.
           d. Step LR scheduler; check early stopping.
           e. Sanity-check on 5 validation examples.
           f. Save per-epoch checkpoint.
      6. Restore best weights; save as color_model_best.pth.
      7. Generate y_pred.csv on the validation split.
    """

    # ------------------------------------------------------------------
    # Hyperparameters — edit these to tune the run
    # ------------------------------------------------------------------
    MAX_EPOCHS        = 30    # upper bound; early stopping will fire sooner
    BATCH_SIZE        = 32    # reduce to 16 if you hit CUDA OOM
    LEARNING_RATE     = 2e-3  # LR for the MLP classifier head
    ENCODER_LR        = 1e-5  # LR for the unfrozen encoder layers — much smaller
                               # to prevent catastrophic forgetting of pretrained
                               # weights.  Rule of thumb: 100–200× smaller than
                               # the head LR.
    UNFREEZE_LAYERS   = 3     # number of final transformer layers to unfreeze in
                               # each encoder.  Increased from 2 → 3 for the full
                               # 212k run — more data means we can safely fine-tune
                               # a larger portion of the encoder without overfitting.
    VAL_RATIO         = 0.1   # 10% of data held out for validation
    TRAIN_THRESHOLD   = 0.5   # sigmoid cutoff for training-phase F1 reporting
    VAL_THRESHOLD     = 0.5   # sigmoid cutoff for validation-phase F1 reporting
                               # raised from 0.3 — at 50k+ samples the model is
                               # confident enough that 0.3 causes over-prediction

    # Early stopping: stop if val loss doesn't improve by min_delta for
    # `es_patience` consecutive epochs.
    # Increased from 5 → 7: in the 100k run, val F1 kept improving for 5 epochs
    # after early stopping fired on loss — giving more patience captures those gains.
    ES_PATIENCE  = 7
    ES_MIN_DELTA = 1e-4

    # ReduceLROnPlateau: halve LR if val loss doesn't improve for
    # `lr_patience` consecutive epochs.
    LR_PATIENCE  = 2
    LR_FACTOR    = 0.5   # new_lr = old_lr * LR_FACTOR
    LR_MIN       = 1e-6  # floor — LR will never drop below this

    # Model IDs — change here if you want to swap encoders
    TEXT_MODEL_ID   = "cl-tohoku/bert-base-japanese-v3"
    VISION_MODEL_ID = "openai/clip-vit-base-patch16"

    # Paths — relative to the repository root
    X_TRAIN_PATH = "data/processed/X_train_processed.csv"
    Y_TRAIN_PATH = "data/processed/y_train_processed.csv"
    IMG_DIR      = "data/images/"
    MODEL_DIR    = "models/"

    # Ensure the models output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # ------------------------------------------------------------------
    # Load tokenizer, image processor, and frozen encoders
    # ------------------------------------------------------------------
    print("\nLoading tokenizer and image processor...")
    tokenizer       = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL_ID)

    print("Loading dual encoder (BERT + CLIP vision)...")
    dual_encoder = DualEncoder(TEXT_MODEL_ID, VISION_MODEL_ID).to(device)

    # Unfreeze the last UNFREEZE_LAYERS transformer layers of each encoder.
    # All other layers remain frozen.  The unfrozen layers will be trained at
    # ENCODER_LR (much smaller than the head LR) via separate param groups below.
    print(f"Unfreezing last {UNFREEZE_LAYERS} encoder layers...")
    dual_encoder.unfreeze_encoder_layers(num_layers=UNFREEZE_LAYERS)

    # train() mode is now correct since the unfrozen encoder layers need
    # Dropout active during training to regularise fine-tuning.
    # We switch back to eval() at the start of each validation phase.
    dual_encoder.train()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    X_tr, X_va, y_tr, y_va, mlb_obj = prepare_split_data(
        X_TRAIN_PATH, Y_TRAIN_PATH, MODEL_DIR, val_ratio=VAL_RATIO
    )

    print(f"\nTrain size: {len(X_tr):,}  |  Val size: {len(X_va):,}")

    # Pre-validate images upfront — removes any corrupt/missing files from
    # the index lists so that __getitem__ never raises a runtime error.
    print("\nValidating training images...")
    train_valid_idx = build_valid_indices(X_tr.reset_index(drop=True), IMG_DIR)
    print("Validating validation images...")
    val_valid_idx   = build_valid_indices(X_va.reset_index(drop=True), IMG_DIR)

    # ------------------------------------------------------------------
    # Datasets and DataLoaders
    # ------------------------------------------------------------------
    train_ds = MultimodalColorDataset(
        X_tr, y_tr, IMG_DIR, tokenizer, image_processor, train_valid_idx
    )
    val_ds = MultimodalColorDataset(
        X_va, y_va, IMG_DIR, tokenizer, image_processor, val_valid_idx
    )

    # IMPORTANT: num_workers=0, pin_memory=False
    # Colab uses a forked subprocess model that raises AssertionError during
    # DataLoader worker cleanup when num_workers > 0.  Setting num_workers=0
    # runs data loading in the main process — slightly slower but error-free.
    # pin_memory=True only benefits GPU transfers when using background workers,
    # so it is disabled here as well.
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # ------------------------------------------------------------------
    # Classifier, loss, optimiser, scheduler, early stopping
    # ------------------------------------------------------------------
    num_classes = len(mlb_obj.classes_)
    classifier  = ColorClassifier(input_dim=1536, num_colors=num_classes).to(device)

    # pos_weight for BCEWithLogitsLoss
    # --------------------------------
    # In a multi-label dataset, most classes are absent for most items (sparse
    # labels).  Without weighting, the model quickly learns to predict "absent"
    # for everything and gets a low loss while producing useless predictions.
    #
    # pos_weight[c] = num_negatives_c / num_positives_c
    #
    # This makes a false negative (missing a present colour) exactly as costly
    # as (pos_weight[c]) false positives for that colour, proportional to how
    # rare it is.  Common colours (e.g. "white") get a lower weight; rare
    # colours (e.g. "gold") get a higher weight.
    #
    # The weight is capped at 10.0 to prevent over-correction on rare classes.
    # At 9k samples, uncapped weights were necessary to learn rare colours at all.
    # At 50k+ samples, rare classes have enough examples that uncapped weights
    # cause the model to over-predict them, producing too many false positives.
    label_counts = y_tr.sum(axis=0).astype(float)          # positives per class
    neg_counts   = len(y_tr) - label_counts                 # negatives per class
    pos_weights  = torch.tensor(
        np.clip(neg_counts / (label_counts + 1e-6), a_min=1.0, a_max=10.0),
        dtype=torch.float32
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Two-group optimiser
    # -------------------
    # The MLP head and the unfrozen encoder layers are trained at very different
    # learning rates.  Adam supports this via parameter groups — each group can
    # have its own lr, weight_decay, etc.
    #
    # Group 1 — classifier head  : lr = LEARNING_RATE (2e-3)
    #   The head is randomly initialised and needs a large LR to learn quickly.
    #
    # Group 2 — unfrozen encoder layers : lr = ENCODER_LR (1e-5)
    #   These layers have carefully pretrained weights.  A tiny LR nudges them
    #   toward colour-relevant representations without overwriting what they
    #   already know (catastrophic forgetting).
    #
    # Frozen encoder layers have requires_grad=False and are automatically
    # excluded from the computation graph — no need to list them separately.
    encoder_params = [p for p in dual_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": classifier.parameters(),  "lr": LEARNING_RATE},
            {"params": encoder_params,            "lr": ENCODER_LR},
        ]
    )

    # ReduceLROnPlateau
    # -----------------
    # Monitors validation loss ("min" mode).  If val loss does not decrease by
    # at least threshold=1e-4 for `LR_PATIENCE` epochs, the LR is multiplied
    # by LR_FACTOR (0.5), halving it.  This allows the model to escape flat
    # loss landscapes without manual LR tuning.
    # min_lr sets a floor so the LR never decays to effectively zero.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
        # Note: verbose=True was removed — it was dropped in PyTorch 2.x.
        # The current LR is already printed each epoch in the summary line.
    )

    # EarlyStopping
    # -------------
    # Monitors val F1 (higher is better) rather than val loss.
    # Val loss can plateau while F1 is still improving — stopping on F1
    # ensures training continues as long as colour predictions get better.
    # ReduceLROnPlateau still uses val loss (separate concern: escaping
    # loss plateaus by reducing LR is independent of when to stop entirely).
    early_stopping = EarlyStopping(
        patience=ES_PATIENCE, min_delta=ES_MIN_DELTA, mode="max"
    )

    # Mixed Precision (AMP)
    # ---------------------
    # autocast runs encoder and head forward passes in float16 on GPU,
    # which roughly doubles throughput on T4/L4/A100 Tensor Cores with
    # negligible effect on accuracy.
    # GradScaler prevents fp16 gradients from underflowing to zero during
    # the backward pass through the classifier head.
    # Both are disabled automatically when running on CPU (use_amp=False).
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)


    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting training (max {MAX_EPOCHS} epochs, "
          f"early stopping patience={ES_PATIENCE})...\n")

    for epoch in range(MAX_EPOCHS):

        # ---- Training phase ----
        # Both the classifier head and the unfrozen encoder layers are set to
        # train() mode, which enables Dropout in both for regularisation.
        classifier.train()
        dual_encoder.train()
        all_preds, all_labels = [], []
        t_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS} [Train]")
        for batch in pbar:
            optimizer.zero_grad()   # clear gradients from previous step

            px     = batch["pixel_values"].to(device)
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            target = batch["label"].to(device)

            # No torch.no_grad() here — the unfrozen encoder layers need
            # gradients to flow back through them during backprop.
            # autocast still applies for fp16 throughput on Tensor Cores.
            with autocast("cuda", enabled=use_amp):
                img_embeds = dual_encoder.get_image_features(px)
                txt_embeds = dual_encoder.get_text_features(ids, mask)
                logits     = classifier(img_embeds, txt_embeds)

                # BCEWithLogitsLoss = sigmoid + binary cross-entropy, computed in
                # a single numerically stable operation.  pos_weight is applied
                # internally to up-weight false negatives for rare classes.
                loss = loss_fn(logits, target)

            # Scaled backward pass — GradScaler multiplies the loss by a large
            # scale factor before backward() to prevent fp16 gradient underflow,
            # then divides back before the optimiser step.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Collect binary predictions for epoch-level F1 calculation.
            # Threshold 0.5 for training reporting (stricter = more precise).
            preds = (torch.sigmoid(logits) > TRAIN_THRESHOLD).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(target.cpu().numpy())
            t_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Micro-averaged F1 across all classes and all samples.
        # "Micro" sums TP/FP/FN across classes before computing F1, which
        # gives common classes more influence — appropriate here because
        # common colours (white, black) are also the most business-important.
        train_f1 = f1_score(
            np.vstack(all_labels), np.vstack(all_preds),
            average="micro", zero_division=0
        )

        # ---- Validation phase ----
        # Both classifier and encoder are set to eval() — disables Dropout
        # in both for deterministic, reproducible validation predictions.
        classifier.eval()
        dual_encoder.eval()
        val_preds, val_labels = [], []
        v_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS} [Val]"):
                px     = batch["pixel_values"].to(device)
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                target = batch["label"].to(device)

                # autocast for val too — no GradScaler needed since there
                # is no backward pass during validation.
                with autocast("cuda", enabled=use_amp):
                    img_embeds = dual_encoder.get_image_features(px)
                    txt_embeds = dual_encoder.get_text_features(ids, mask)
                    logits     = classifier(img_embeds, txt_embeds)
                v_loss    += loss_fn(logits, target).item()

                # Threshold 0.3 for validation — lower than training to favour
                # recall, catching more true positives at the cost of some
                # precision.  Colour tagging generally tolerates false positives
                # better than missed colours.
                preds = (torch.sigmoid(logits) > VAL_THRESHOLD).int().cpu().numpy()
                val_preds.append(preds)
                val_labels.append(target.cpu().numpy())

        avg_val_loss = v_loss / len(val_loader)
        val_f1 = f1_score(
            np.vstack(val_labels), np.vstack(val_preds),
            average="micro", zero_division=0
        )
        current_lr_head    = optimizer.param_groups[0]["lr"]
        current_lr_encoder = optimizer.param_groups[1]["lr"]

        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"Train Loss : {t_loss / len(train_loader):.4f}  |  Train F1 : {train_f1:.4f}")
        print(f"Val   Loss : {avg_val_loss:.4f}  |  Val   F1 : {val_f1:.4f}  "
              f"|  LR head : {current_lr_head:.2e}  |  LR enc : {current_lr_encoder:.2e}")

        # Step the LR scheduler — checks whether val loss has stagnated and
        # reduces the LR if so.  Must be called after validation, before
        # the next training epoch.
        scheduler.step(avg_val_loss)

        # Save per-epoch checkpoint — both classifier head and encoder
        # since both are being updated during training.
        checkpoint = {
            "classifier":   classifier.state_dict(),
            "dual_encoder": dual_encoder.state_dict(),
        }
        ckpt_path = os.path.join(MODEL_DIR, f"color_model_ep{epoch + 1}.pth")
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Check early stopping — pass val_f1 (higher = better).
        # ReduceLROnPlateau still receives avg_val_loss separately above.
        if early_stopping(val_f1, classifier, encoder=dual_encoder):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    # ------------------------------------------------------------------
    # Restore best weights and save as the definitive model file
    # ------------------------------------------------------------------
    # This ensures we always use the checkpoint with the lowest val loss,
    # not the weights from the last training epoch (which may be overfit).
    early_stopping.restore_best_weights(classifier, encoder=dual_encoder)
    best_checkpoint = {
        "classifier":   classifier.state_dict(),
        "dual_encoder": dual_encoder.state_dict(),
    }
    best_path = os.path.join(MODEL_DIR, "color_model_best.pth")
    torch.save(best_checkpoint, best_path)
    print(f"\nBest model saved to '{best_path}'.")

    # ------------------------------------------------------------------
    # Generate y_pred.csv on the validation split
    # ------------------------------------------------------------------
    generate_predictions(
        classifier      = classifier,
        dual_encoder    = dual_encoder,
        df_x            = X_va,
        img_dir         = IMG_DIR,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        mlb             = mlb_obj,
        device          = device,
        threshold       = VAL_THRESHOLD,
        min_preds       = 1,
        out_path        = os.path.join(MODEL_DIR, "y_pred_training.csv"),
    )


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    train()