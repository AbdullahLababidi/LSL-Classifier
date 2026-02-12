from pathlib import Path
import torch
from train_sign_model import DualInputMobileNet
from sign_dataset import CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"

MODEL_PATH = ROOT_DIR / "best_model.pth"
ONNX_OUT   = ROOT_DIR / "best_model.onnx"

device = "cpu"
model = DualInputMobileNet(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Dummy inputs: two images, 224x224
x1 = torch.randn(1, 3, 256, 256, device=device)
x2 = torch.randn(1, 3, 256, 256, device=device)


torch.onnx.export(
    model,
    (x1, x2),
    ONNX_OUT,
    input_names=["img1", "img2"],
    output_names=["logits"],
    opset_version=12,
    dynamic_axes=None
)

print("Saved:", ONNX_OUT)
