import os
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

# ========== CONFIG ==========
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"

INPUT_DIR = Path(r"D:\Capstone\Pictures ALL") #Needs the original Data
OUTPUT_DIR = ROOT_DIR
TARGET_SHORT_SIDE = 256   # only resize shortest side to this
# ============================


def resize_keep_ratio(in_path: Path, out_path: Path):
    try:
        img = Image.open(in_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        w, h = img.size
        short_side = min(w, h)

        # Compute scale factor
        scale = TARGET_SHORT_SIDE / short_side
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize without cropping or padding
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Save to output directory, mirroring the folder structure
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=95)

    except Exception as e:
        print(f"[ERROR] Could not process {in_path}: {e}")


def main():
    # Collect all jpg/jpeg files
    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        all_images.extend(INPUT_DIR.rglob(ext))

    print(f"Found {len(all_images)} images to resize.")
    for img_path in tqdm(all_images):
        relative = img_path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / relative
        resize_keep_ratio(img_path, out_path)

    print("Done! Resized images saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
