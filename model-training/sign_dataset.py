from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import ImageOps

# ========== CONFIG ==========
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"

TRAIN_CSV = ROOT_DIR / "train.csv"
VAL_CSV   = ROOT_DIR / "val.csv"
TEST_CSV  = ROOT_DIR / "test.csv"

# All 10 words (exactly as they appear in the CSV)
CLASSES = [
    "Bye",
    "Good Morning",
    "How are you",
    "Ilhamdillah",
    "Me",
    "No",
    "Please",
    "Sorry",
    "Thank you",
    "Yes",
]
WORD_TO_IDX = {w: i for i, w in enumerate(CLASSES)}
IDX_TO_WORD = {i: w for w, i in WORD_TO_IDX.items()}
# ============================


class SignWordDataset(Dataset):
    """
    Dual-input dataset:
      - img1: path in column 'img1'
      - img2: path in column 'img2'
      - label: integer index of 'word'
    """
    def __init__(self, csv_path: Path, word_to_idx: dict, transform: Optional[Callable] = None):
        self.df = pd.read_csv(csv_path)
        self.img1_paths = self.df["img1"].tolist()
        self.img2_paths = self.df["img2"].tolist()
        self.labels = [word_to_idx[w] for w in self.df["word"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img1 = self._load_image(self.img1_paths[idx])
        img2 = self._load_image(self.img2_paths[idx])
        label = self.labels[idx]

        if self.transform is not None:
            # Only using non-random transforms for now so it's fine to call separately
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


def pad_to_square(img):
    """
    Pad the PIL image to a square using white borders, without cropping.
    """
    w, h = img.size
    max_side = max(w, h)
    pad_left  = (max_side - w) // 2
    pad_top   = (max_side - h) // 2
    pad_right = max_side - w - pad_left
    pad_bottom = max_side - h - pad_top

    # Add padding (left, top, right, bottom)
    img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))
    return img


def get_transforms():
    """
    1. Pad to square so all images become e.g. 341x341 (no cropping).
    2. Resize to 256x256 (uniform scaling, still no cropping).
    3. Convert to tensor + normalize.
    """
    return T.Compose([
        T.Lambda(pad_to_square),
        T.Resize((256, 256)),  # now safe: we're resizing a square, not cropping
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])



def create_dataloaders(batch_size: int = 32, num_workers: int = 0):
    transform = get_transforms()

    train_ds = SignWordDataset(TRAIN_CSV, WORD_TO_IDX, transform=transform)
    val_ds   = SignWordDataset(VAL_CSV,   WORD_TO_IDX, transform=transform)
    test_ds  = SignWordDataset(TEST_CSV,  WORD_TO_IDX, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=8, num_workers=0)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    batch = next(iter(train_loader))
    img1, img2, labels = batch

    print("img1 shape:", img1.shape)   # expected: [B, 3, H, W]
    print("img2 shape:", img2.shape)   # same
    print("labels shape:", labels.shape)
    print("Example labels:", labels[:8])
