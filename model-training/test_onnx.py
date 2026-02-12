from pathlib import Path
import onnxruntime as ort
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # goes up to LSL-Classifier/
ROOT_DIR = REPO_ROOT / "data"

MODEL = ROOT_DIR / "best_model.onnx"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
print("Inputs:", [i.name for i in sess.get_inputs()])
print("Input shapes:", [i.shape for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])

# dummy 2 images (batch=1, 3x224x224)
x1 = np.random.rand(1, 3, 256, 256).astype(np.float32)
x2 = np.random.rand(1, 3, 256, 256).astype(np.float32)

out = sess.run(None, {sess.get_inputs()[0].name: x1,
                      sess.get_inputs()[1].name: x2})
print("Output shape:", out[0].shape)
print("OK ✅")
