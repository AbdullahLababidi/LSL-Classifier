from pathlib import Path
import pandas as pd

# ========== CONFIG ==========
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/

ROOT_DIR = REPO_ROOT / "data"
OUTPUT_CSV = REPO_ROOT/ "images_index.csv"
# ============================

def main():
    rows = []

    for word_dir in sorted(ROOT_DIR.iterdir()):
        if not word_dir.is_dir():
            continue

        word = word_dir.name

        for part_name in ["Sign 1", "Sign 2"]:
            part_dir = word_dir / part_name
            if not part_dir.exists():
                continue

            # FIXED: only read .jpg once
            images = [p for p in part_dir.iterdir() if p.suffix.lower() == ".jpg"]
            images = sorted(images)

            for img_path in images:
                rows.append({
                    "word": word,
                    "part": part_name,
                    "filepath": str(img_path),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved index CSV to: {OUTPUT_CSV}\n")

    # Summary
    summary = df.groupby(["word","part"]).size().reset_index(name="count")
    print(summary)


if __name__ == "__main__":
    main()
