# EPI Hardhat Detector

Real-time detection of PPE (capacete/hardhat) via webcam using a pretrained YOLO model. Includes optional fine-tuning pipeline focused on color invariance so helmets of any color are detected reliably.

## Features
- Real-time webcam detection with bounding boxes and labels
- Uses Ultralytics YOLOv8/YOLOv5 pretrained weights
- Fine-tuning script with color jitter/grayscale augmentation to promote color invariance
- Simple dataset folder structure and config

## Quick start
Recommended: Python 3.10 or 3.11 on Windows for prebuilt wheels. Python 3.13 may try to compile NumPy from source and fail without Visual Studio C++ Build Tools.

1. Create a virtual environment and install dependencies
2. Run the webcam app

```powershell
# 1) Create venv (use Python 3.11 if available)
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1

# 2) Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Run webcam detector
python app.py

# Optional: run on an image file and save result
python app.py --source .\data\images\sample.jpg --save --out .\outputs\result.jpg

# Optional: run on a video file
python app.py --source .\data\videos\sample.mp4
```

Press `Q` to quit.

## Training (optional)
If you have a dataset with `hardhat` labels, you can fine-tune to improve color invariance.

```powershell
# Activate env
.\.venv\Scripts\Activate.ps1

# Train
python scripts\train.py --data data\dataset.yaml --epochs 50 --batch 16 --img 640 --weights yolov8n.pt
```

The training script applies strong color augmentations including grayscale probability to enforce invariance.

## Dataset structure
```
data/
  dataset.yaml          # YOLO data config
  images/
    train/
    val/
  labels/
    train/
    val/
```

`dataset.yaml` example is provided and assumes a single class `hardhat`.

## Notes
- Requires a working webcam.
- Tested with Python 3.10/3.11.
- If you prefer CPU only, it will still run but slower.
 - For best results, fine-tune a model to the `hardhat` class and provide it via `--weights path/to/hardhat.pt`.

### Troubleshooting
- If installation fails on NumPy with Python 3.13, prefer Python 3.11 (py -3.11) or install the Microsoft C++ Build Tools and ensure a compatible NumPy wheel is available.
