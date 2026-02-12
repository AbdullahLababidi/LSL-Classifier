from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from sign_dataset import create_dataloaders, CLASSES
from train_sign_model import DualInputMobileNet  # reuse the same class


# ========== CONFIG ==========
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"
SAVE_PATH = ROOT_DIR / "best_model.pth"
# ============================


def evaluate_on_test(model, loader: DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            outputs = model(img1, img2)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print("\n=== Test Set Evaluation ===")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get only test loader (we don't care about train/val here)
    _, _, test_loader = create_dataloaders(batch_size=32, num_workers=0)

    # Build model and load weights
    model = DualInputMobileNet(num_classes=len(CLASSES)).to(device)
    state_dict = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # Evaluate
    evaluate_on_test(model, test_loader, device)


if __name__ == "__main__":
    main()
