import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tkinter import Tk, filedialog

from sign_dataset import get_transforms, CLASSES
from train_sign_model import DualInputMobileNet


# ========== CONFIG ==========
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"
MODEL_PATH = ROOT_DIR / "best_model.pth"
# ============================


def load_image(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)   # [1, 3, H, W]


def choose_images():
    Tk().withdraw()  # Hide empty Tkinter window

    print("\nSelect **ONE** image for 1-sign words, or **TWO** images for 2-sign words.")

    files = filedialog.askopenfilenames(
        title="Select 1 or 2 images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if len(files) == 0:
        print("No images selected.")
        return None, None, False

    if len(files) == 1:
        print("[INFO] 1 image selected → using it for both inputs.")
        return files[0], files[0], False

    if len(files) == 2:
        print("[INFO] 2 images selected.")
        return files[0], files[1], True

    print("[ERROR] Please select only 1 or 2 images.")
    return None, None, False


def main():
    # Ask user to choose images
    img1_path, img2_path, two_imgs = choose_images()
    if img1_path is None:
        return

    img1_path = Path(img1_path)
    img2_path = Path(img2_path)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DualInputMobileNet(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = get_transforms()

    # Load images
    img1 = load_image(img1_path, transform).to(device)
    img2 = load_image(img2_path, transform).to(device)

    # Predict
    with torch.no_grad():
        logits = model(img1, img2)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_word = CLASSES[pred_idx]

    # Output
    print("\n=== Prediction ===")
    print(f"Image 1: {img1_path}")
    if two_imgs:
        print(f"Image 2: {img2_path}")
    else:
        print("[INFO] 1-image mode → img1 used as both inputs.")

    print(f"\nPredicted Word: {pred_word}\n")

    print("Class Probabilities:")
    for w, p in zip(CLASSES, probs):
        print(f"{w:13s}: {float(p):.4f}")


if __name__ == "__main__":
    main()
