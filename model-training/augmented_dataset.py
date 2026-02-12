from pathlib import Path
import os
import cv2
import numpy as np
import random

# ---- YOUR FOLDER PATHS ----
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"

INPUT_DIR = ROOT_DIR
OUTPUT_DIR = REPO_ROOT / "Augmented data"

# The single-sign words you want to augment
CLASSES = ["Good Morning", "Me", "Please", "Yes"]

# Make output directories
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# ---- Augmentation Functions ----
def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def rotate(img):
    angle = random.randint(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def change_brightness(img):
    factor = random.uniform(0.7, 1.3)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

AUGMENTATIONS = [add_noise, rotate, change_brightness]


# ---- Process Each Folder ----
for cls in CLASSES:
    input_path = os.path.join(INPUT_DIR, cls, "Sign 1")
    output_path = os.path.join(OUTPUT_DIR, cls)

    images = os.listdir(input_path)

    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # pick ONE augmentation randomly
        aug_func = random.choice(AUGMENTATIONS)
        aug_img = aug_func(img)

        # save augmented image
        save_name = img_name.replace(".jpg", "").replace(".png", "")
        save_name = f"{save_name}_aug.jpg"

        cv2.imwrite(os.path.join(output_path, save_name), aug_img)

print("Done! 1 augmented image created for each original image.")
