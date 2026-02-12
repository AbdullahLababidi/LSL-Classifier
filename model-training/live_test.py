import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import time

from sign_dataset import get_transforms, CLASSES
from train_sign_model import DualInputMobileNet


# ===================== CONFIG ======================
REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"
MODEL_PATH = ROOT_DIR / "best_model.pth"
CAPTURE_DELAY = 1.0   # seconds between image1 and image2
# ==================================================


# Convert OpenCV → PIL → Transform → Tensor
def preprocess_frame(frame, transform):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0)   # shape [1,3,H,W]
    return tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = DualInputMobileNet(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = get_transforms()

    # Start webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Camera not found.")
        return

    print("Press Q to quit.\n")

    frame1 = None
    frame2 = None
    last_capture_time = time.time()
    capture_stage = 1  # 1 = capture image1, 2 = capture image2

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Camera read error")
            break

        # Display live feed
        cv2.imshow("Live Sign Recognition - Press Q to quit", frame)

        current_time = time.time()

        # ====== CAPTURE IMAGE 1 ======
        if capture_stage == 1 and current_time - last_capture_time >= CAPTURE_DELAY:
            frame1 = frame.copy()
            print("[INFO] Captured Image 1")
            capture_stage = 2
            last_capture_time = time.time()

        # ====== CAPTURE IMAGE 2 ======
        elif capture_stage == 2 and current_time - last_capture_time >= CAPTURE_DELAY:
            frame2 = frame.copy()
            print("[INFO] Captured Image 2")
            capture_stage = 1
            last_capture_time = time.time()

            # ====== Predict now that we have 2 frames ======
            img1 = preprocess_frame(frame1, transform).to(device)
            img2 = preprocess_frame(frame2, transform).to(device)

            with torch.no_grad():
                logits = model(img1, img2)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = probs.argmax().item()
                pred_word = CLASSES[pred_idx]

            print("\n=== PREDICTION ===")
            print(f"Predicted Word: {pred_word}")
            print("---------------------------")
            for w, p in zip(CLASSES, probs):
                print(f"{w:15s}: {float(p):.4f}")
            print("---------------------------\n")

        # Quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
