"""
CLIP Zero-Shot Color Classification.

Nutzt CLIP um aus Produktbildern Farb-Wahrscheinlichkeiten abzuleiten,
ohne Training – rein ueber Text-Bild Aehnlichkeit.

Nutzung:
    from src.models.clip_model import clip_zero_shot_scores
    probs = clip_zero_shot_scores(df_x, image_dir="data/images")
"""
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLOR_LABELS, NUM_LABELS, CLIP_CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Prompt-Templates fuer bessere Zero-Shot Performance ──
COLOR_PROMPTS = [
    "a photo of a {} colored product",
    "a {} item of clothing",
    "a product that is {} in color",
    "this is a {} product",
]


def _build_text_features(model, tokenizer):
    """Erstellt gemittelte Text-Embeddings fuer alle Farben."""
    all_features = []

    for color in COLOR_LABELS:
        prompts = [tpl.format(color.lower()) for tpl in COLOR_PROMPTS]
        tokens = tokenizer(prompts).to(DEVICE)

        with torch.no_grad():
            features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
            # Mittelwert ueber alle Prompts pro Farbe
            mean_feature = features.mean(dim=0)
            mean_feature /= mean_feature.norm()

        all_features.append(mean_feature)

    # (NUM_LABELS, embed_dim)
    return torch.stack(all_features)


def clip_zero_shot_scores(df_x, image_dir=None, config=None):
    """
    Berechnet CLIP Zero-Shot Scores fuer alle Produkte.

    Args:
        df_x: DataFrame mit 'image_file_name' Spalte
        image_dir: Pfad zu den Bildern
        config: CLIP-Konfiguration (optional)

    Returns:
        np.ndarray: (n_samples, NUM_LABELS) Probability-Matrix
    """
    config = {**CLIP_CONFIG, **(config or {})}
    image_dir = Path(image_dir) if image_dir else config['image_dir']

    try:
        import open_clip
        from PIL import Image
    except ImportError:
        print("WARNUNG: open_clip nicht installiert. Nutze Dummy-Scores.")
        return _dummy_scores(len(df_x))

    print(f"\n{'='*60}")
    print(f"CLIP Zero-Shot Scoring")
    print(f"  Model:  {config['model_name']}")
    print(f"  Device: {DEVICE}")
    print(f"  Images: {image_dir}")
    print(f"{'='*60}\n")

    # ── Modell laden ──
    model, _, preprocess = open_clip.create_model_and_transforms(
        config['model_name'],
        pretrained=config['pretrained'],
        device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(config['model_name'])
    model.eval()

    # ── Text-Features (einmalig) ──
    text_features = _build_text_features(model, tokenizer)

    # ── Bild-Features batch-weise ──
    all_probs = []
    batch_size = config.get('batch_size', 64)
    image_files = df_x['image_file_name'].tolist()

    for i in tqdm(range(0, len(image_files), batch_size), desc="CLIP Scoring"):
        batch_files = image_files[i:i+batch_size]
        images = []

        for fname in batch_files:
            img_path = image_dir / fname
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                images.append(preprocess(img))
            else:
                # Fallback: schwarzes Bild wenn Datei fehlt
                images.append(torch.zeros(3, 224, 224))

        image_batch = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Cosine Similarity -> Softmax Probabilities
            similarity = (100.0 * image_features @ text_features.T)
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()

        all_probs.append(probs)

    prob_matrix = np.vstack(all_probs)
    print(f"  CLIP Scoring fertig: {prob_matrix.shape}")
    return prob_matrix


def _dummy_scores(n_samples):
    """
    Fallback wenn CLIP nicht verfuegbar ist.
    Gibt gleichverteilte Pseudo-Scores zurueck.
    """
    print("  -> Dummy-Scores (uniform) werden verwendet.")
    rng = np.random.RandomState(42)
    raw = rng.dirichlet(np.ones(NUM_LABELS), size=n_samples)
    return raw


if __name__ == "__main__":
    import pandas as pd
    from src.config import DATA_DIR

    df_x = pd.read_csv(DATA_DIR / "X_train.csv")
    probs = clip_zero_shot_scores(df_x.head(10))
    print(f"\nBeispiel-Scores (erstes Produkt):")
    for color, score in zip(COLOR_LABELS, probs[0]):
        if score > 0.05:
            print(f"  {color:>18}: {score:.3f}")
