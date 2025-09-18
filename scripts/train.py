import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO for hardhat detection with color invariance")
    p.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="Base weights to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="hardhat-color-invariant")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    # Recommended augmentation config to promote color invariance.
    # Ultralytics accepts 'cfg' style dicts via overrides. Key options include:
    overrides = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.img,
        'batch': args.batch,
        'project': args.project,
        'name': args.name,
        'patience': 25,
    # Strong color aug
    'hsv_h': 0.015,      # hue augmentation (default ~0.015)
    'hsv_s': 0.9,        # increase saturation range
    'hsv_v': 0.7,        # increase value range
    'scale': 0.5,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'copy_paste': 0.2,
    # MixUp can help generalization
    'mixup': 0.2,
    # Enable pretrained backbone and freeze partially for stability
    'pretrained': True,
    # Optionally freeze backbone early layers (reduce overfitting)
    # 'freeze': 10,
    }

    model.train(**overrides)

    # Validate trained model
    model.val(data=args.data, imgsz=args.img)


if __name__ == "__main__":
    main()
