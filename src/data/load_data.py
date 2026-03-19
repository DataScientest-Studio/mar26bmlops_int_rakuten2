from pathlib import Path
import ast
import pandas as pd


def main():
    base_path = Path(__file__).resolve().parents[2]
    raw_path = base_path / "data" / "raw"
    processed_path = base_path / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    x_train_path = raw_path / "X_train.csv"
    y_train_path = raw_path / "y_train.csv"
    x_test_path = raw_path / "X_test.csv"

    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(x_test_path)

    # Drop auto index column if present
    for df in [X_train, y_train, X_test]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # Fill missing text fields
    X_train["item_name"] = X_train["item_name"].fillna("")
    X_train["item_caption"] = X_train["item_caption"].fillna("")
    X_test["item_name"] = X_test["item_name"].fillna("")
    X_test["item_caption"] = X_test["item_caption"].fillna("")

    # Combine text
    X_train["combined_text"] = (
        X_train["item_name"].astype(str) + " " + X_train["item_caption"].astype(str)
    ).str.strip()

    X_test["combined_text"] = (
        X_test["item_name"].astype(str) + " " + X_test["item_caption"].astype(str)
    ).str.strip()

    # Parse labels from string to list
    y_train["color_tags_list"] = y_train["color_tags"].apply(ast.literal_eval)

    print("\nShapes after cleaning")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")

    print("\nColumns in cleaned X_train")
    print(X_train.columns.tolist())

    print("\nColumns in cleaned y_train")
    print(y_train.columns.tolist())

    print("\nMissing values after cleaning")
    print(X_train.isna().sum())
    print(y_train.isna().sum())

    print("\nSample parsed labels")
    print(y_train["color_tags_list"].head(10).tolist())

    print("\nNumber of labels per sample (first 10)")
    print(y_train["color_tags_list"].apply(len).head(10).tolist())

    print("\nSample combined text")
    print(X_train["combined_text"].head(3).tolist())

    unique_colors = sorted(
        set(color for labels in y_train["color_tags_list"] for color in labels)
    )
    print("\nUnique color labels")
    print(unique_colors)

    print(f"\nNumber of unique color labels: {len(unique_colors)}")

    label_counts = (
        y_train["color_tags_list"]
        .explode()
        .value_counts()
        .sort_values(ascending=False)
    )

    print("\nTop 15 most frequent labels")
    print(label_counts.head(15))

    # Save processed files
    X_train.to_csv(processed_path / "X_train_processed.csv", index=False)
    y_train.to_csv(processed_path / "y_train_processed.csv", index=False)
    X_test.to_csv(processed_path / "X_test_processed.csv", index=False)

    label_counts.to_csv(processed_path / "label_counts.csv", header=True)

    print("\nSaved processed files to:")
    print(processed_path / "X_train_processed.csv")
    print(processed_path / "y_train_processed.csv")
    print(processed_path / "X_test_processed.csv")
    print(processed_path / "label_counts.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()