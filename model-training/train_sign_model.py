from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from sign_dataset import create_dataloaders, CLASSES  # import from your previous file


# ========== CONFIG ==========
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"
SAVE_PATH = ROOT_DIR / "best_model.pth"

NUM_CLASSES = len(CLASSES)
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
# ============================


class DualInputMobileNet(nn.Module):
    """
    Two-image input model:
      - Shared MobileNetV2 backbone for img1 and img2
      - Global average pool -> feature vectors
      - Concatenate -> classification head -> 10 classes
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # Pretrained MobileNetV2
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # backbone = models.mobilenet_v2(weights=None)  # <- use this if weights download fails

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # global avg pool
        feature_dim = 1280  # MobileNetV2 output channels

        # Classification head after concatenating features from img1 + img2
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # [B, 1280]
        return x

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        f1 = self.forward_single(img1)
        f2 = self.forward_single(img2)
        fused = torch.cat([f1, f2], dim=1)  # [B, 2560]
        out = self.classifier(fused)
        return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for img1, img2, labels in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_on_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
):
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
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=BATCH_SIZE, num_workers=0)

    # Model
    model = DualInputMobileNet(num_classes=NUM_CLASSES).to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n==== Epoch {epoch}/{EPOCHS} ====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train   | loss: {train_loss:.4f}  acc: {train_acc:.4f}")

        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Val     | loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[INFO] New best model saved with val_acc={val_acc:.4f}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {SAVE_PATH}")

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    evaluate_on_test(model, test_loader, device)


if __name__ == "__main__":
    main()
