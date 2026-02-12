import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ========== CONFIG ==========
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/

ROOT_DIR = REPO_ROOT / "data"

INDEX_CSV = ROOT_DIR / "images_index.csv"  # already created before
BLANK_IMAGE_PATH = ROOT_DIR / "blank.jpg"  # same as before (or recreated)
TRAIN_PAIRS_CSV = ROOT_DIR / "train.csv"   # will overwrite old ones
VAL_PAIRS_CSV   = ROOT_DIR / "val.csv"
TEST_PAIRS_CSV  = ROOT_DIR / "test.csv"

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15   # remaining goes to test
RANDOM_SEED = 42
# ============================


def ensure_blank_image():
    """Create or reuse a blank (white) image for 1-sign words."""
    if not BLANK_IMAGE_PATH.exists():
        blank = Image.new("RGB", (256, 256), (255, 255, 255))
        blank.save(BLANK_IMAGE_PATH, "JPEG")
        print(f"[INFO] Created blank image: {BLANK_IMAGE_PATH}")
    else:
        print(f"[INFO] Using existing blank image: {BLANK_IMAGE_PATH}")


def split_images_strict(df: pd.DataFrame):
    """
    For each (word, part) group, assign each image to exactly one split:
    train / val / test with ratios ~70/15/15.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    split_labels = []

    # We'll fill this list in the same order as df
    split_labels = [None] * len(df)

    # Group by word and part ("Sign 1", "Sign 2")
    for (word, part), group in df.groupby(["word", "part"]):
        idxs = group.index.to_list()
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # rest goes to test
        n_test = n - n_train - n_val

        train_idxs = idxs[:n_train]
        val_idxs   = idxs[n_train:n_train + n_val]
        test_idxs  = idxs[n_train + n_val:]

        for i in train_idxs:
            split_labels[i] = "train"
        for i in val_idxs:
            split_labels[i] = "val"
        for i in test_idxs:
            split_labels[i] = "test"

        print(f"[SPLIT] {word} / {part}: total={n}, train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}")

    df["split"] = split_labels
    return df


def build_pairs_for_split(df_split: pd.DataFrame, split_name: str):
    """
    Build ONE pair per image for each split:
      - If word has Sign 2 in this split: 1-to-1 pairing (min(len(sign1), len(sign2)))
      - If word has only Sign 1: pair with blank.
    """
    rng = random.Random(RANDOM_SEED)
    pairs = []

    for word, group_word in df_split.groupby("word"):
        sign1_paths = group_word[group_word["part"] == "Sign 1"]["filepath"].tolist()
        sign2_paths = group_word[group_word["part"] == "Sign 2"]["filepath"].tolist()

        rng.shuffle(sign1_paths)
        rng.shuffle(sign2_paths)

        if len(sign2_paths) == 0:
            # 1-sign word: each Sign 1 image gets paired with blank
            for p1 in sign1_paths:
                pairs.append({
                    "img1": p1,
                    "img2": str(BLANK_IMAGE_PATH),
                    "word": word
                })
        else:
            # 2-sign word: 1-to-1 pairing
            n = min(len(sign1_paths), len(sign2_paths))
            if n == 0:
                continue
            for i in range(n):
                pairs.append({
                    "img1": sign1_paths[i],
                    "img2": sign2_paths[i],
                    "word": word
                })

    pairs_df = pd.DataFrame(pairs)
    print(f"[{split_name.upper()}] total pairs: {len(pairs_df)}")
    print(pairs_df["word"].value_counts())
    return pairs_df


def main():
    ensure_blank_image()

    df = pd.read_csv(INDEX_CSV)

    # 1) Strict image-level split
    df = split_images_strict(df)

    # Quick sanity check: no overlap in filepaths across splits
    train_files = set(df[df["split"] == "train"]["filepath"])
    val_files   = set(df[df["split"] == "val"]["filepath"])
    test_files  = set(df[df["split"] == "test"]["filepath"])

    print("\n[CHECK] Overlap between splits:")
    print("Train ∩ Val:", len(train_files & val_files))
    print("Train ∩ Test:", len(train_files & test_files))
    print("Val ∩ Test:", len(val_files & test_files))

    # 2) Build pairs per split
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    train_pairs = build_pairs_for_split(train_df, "train")
    val_pairs   = build_pairs_for_split(val_df, "val")
    test_pairs  = build_pairs_for_split(test_df, "test")

    # 3) Save to CSV (overwriting old ones)
    train_pairs.to_csv(TRAIN_PAIRS_CSV, index=False)
    val_pairs.to_csv(VAL_PAIRS_CSV, index=False)
    test_pairs.to_csv(TEST_PAIRS_CSV, index=False)

    print("\n[INFO] Saved strict pairs CSVs:")
    print("  Train:", TRAIN_PAIRS_CSV)
    print("  Val:  ", VAL_PAIRS_CSV)
    print("  Test: ", TEST_PAIRS_CSV)


if __name__ == "__main__":
    main()
