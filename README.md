Lebanese Sign Language (LSL) Recognition System
================================================

This project implements an image-based Lebanese Sign Language (LSL) 
classification system consisting of:

- A PyTorch training pipeline
- Export to ONNX format
- An Android application performing real-time inference using ONNX Runtime

The system enables training a deep learning model in Python and deploying it 
to a mobile Android application for live sign recognition.


------------------------------------------------------------
Repository Structure
------------------------------------------------------------

LSL-Classifier/
│
├── android-app/              Android Studio project (Kotlin + ONNX Runtime)
├── model-training/           Python training and evaluation scripts
│   ├── train_sign_model.py
│   ├── export_to_onnx.py
│   ├── eval_sign_model.py
│   ├── test_onnx.py
│   ├── predict_word.py
│   ├── live_test.py
│   ├── sign_dataset.py
│   ├── augmented_dataset.py
│   ├── resize_image.py
│   └── requirements.txt
│
├── data/                     Processed dataset + model artifacts
│   ├── (10 class folders)
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── best_model.pth
│   └── best_model.onnx
│
└── README.txt


------------------------------------------------------------
Dataset
------------------------------------------------------------

The original raw dataset (~14GB) used during preprocessing is NOT included 
in this repository.

This repository contains:
- The final processed dataset (resized and augmented)
- Train / Validation / Test splits
- CSV index file
- Final trained model files

This ensures reproducibility of the reported results without requiring the 
full raw dataset.

The script resize_image.py requires the original dataset and is included 
for documentation purposes only.


------------------------------------------------------------
Python Training Pipeline
------------------------------------------------------------

Environment Setup:

cd model-training
pip install -r requirements.txt


Train the Model:

python train_sign_model.py

This script:
- Loads the dataset
- Performs training
- Saves the best performing model (best_model.pth)


Evaluate the Model:

python eval_sign_model.py


Export Model to ONNX:

python export_to_onnx.py

This generates:
best_model.onnx


Test ONNX Model:

python test_onnx.py


------------------------------------------------------------
Android Application
------------------------------------------------------------

The Android application performs real-time inference using ONNX Runtime.

Requirements:
- Android Studio (latest stable recommended)
- Android SDK properly configured
- Physical device or emulator

Run the App:
1. Open the android-app/ folder in Android Studio
2. Allow Gradle to sync
3. Run the app on a device or emulator

The app captures camera input and performs inference using the exported 
ONNX model.


------------------------------------------------------------
System Workflow
------------------------------------------------------------

1. Images are resized and augmented (offline preprocessing).
2. Dataset is organized into class folders.
3. Model is trained using PyTorch.
4. Best model is exported to ONNX format.
5. Android app loads ONNX model.
6. Real-time inference is performed on mobile device.


------------------------------------------------------------
Model Files
------------------------------------------------------------

- best_model.pth  -> PyTorch checkpoint
- best_model.onnx -> ONNX model used for Android inference


------------------------------------------------------------
Technologies Used
------------------------------------------------------------

- Python
- PyTorch
- ONNX
- ONNX Runtime
- OpenCV
- Scikit-learn
- Kotlin
- Android Studio
- CameraX


------------------------------------------------------------
Author
------------------------------------------------------------

Abdullah Lababidi
Capstone Project
Abdullah Lababidi
Capstone Project
