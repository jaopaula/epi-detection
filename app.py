import cv2
import sys
import time
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not installed. Please run: pip install -r requirements.txt", file=sys.stderr)
    raise

"""
Real-time hardhat detector using Ultralytics YOLO.

Tip: pass --weights to use a helmet-specific model (e.g., weights/hardhat.pt).
If omitted, defaults to a general model (yolov8n.pt), which may not include a
hardhat class by default.
"""

# Model selection: lightweight default; you can switch to 'yolov8s.pt' for better accuracy.
DEFAULT_WEIGHTS = 'yolov8n.pt'


def load_model(weights_path: str | None = None, device: str | None = None) -> YOLO:
    weights = weights_path or DEFAULT_WEIGHTS
    model = YOLO(weights)
    # Device hint ("cpu", "cuda", "0", etc.) is respected in predict call; storing for reference only
    model.overrides = getattr(model, 'overrides', {}) or {}
    if device:
        model.overrides['device'] = device
    return model


def draw_detections(frame, results, conf_threshold=0.3):
    h, w = frame.shape[:2]
    if not results or len(results) == 0:
        return frame

    names = results[0].names  # class names dict

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            c = float(box.conf.item()) if box.conf is not None else 0.0
            if c < conf_threshold:
                continue
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            x1, y1, x2, y2 = map(int, xyxy)

            color = (0, 255, 0)  # green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {c:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def parse_args():
    ap = argparse.ArgumentParser(description="EPI Hardhat Detector (YOLO)")
    ap.add_argument('--weights', type=str, default=None, help="Path to .pt weights (helmet-specific recommended)")
    ap.add_argument('--conf', type=float, default=0.35, help="Confidence threshold")
    ap.add_argument('--device', type=str, default=None, help="Device: cpu, cuda, or index like 0")
    ap.add_argument('--source', type=str, default="0", help="Webcam index (e.g., 0) or path to image/video file")
    ap.add_argument('--save', action='store_true', help="Save output when running on an image path")
    ap.add_argument('--out', type=str, default="outputs/result.jpg", help="Output image path when using --save")
    ap.add_argument('--grayscale', action='store_true', help="Convert frames to grayscale (3-channel) before inference")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load model
    # Ajuste automático para usar o modelo mais recente se o caminho padrão não existir
    import os
    weights_path = args.weights
    if not os.path.exists(weights_path):
        # Tenta usar o modelo mais recente encontrado
        alt_path = 'runs/train/hardhat-color-invariant3/weights/best.pt'
        if os.path.exists(alt_path):
            print(f"Arquivo de pesos não encontrado em {weights_path}, usando {alt_path}")
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Nenhum arquivo de pesos encontrado. Verifique o treinamento.")
    model = load_model(weights_path, args.device)

    # Source handling: numeric string -> webcam, else treat as path
    source_str = str(args.source)
    if source_str.isdigit():
        cam_index = int(source_str)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Could not open webcam index {cam_index}.", file=sys.stderr)
            sys.exit(1)
    else:
        p = Path(source_str)
        if not p.exists():
            print(f"Source path not found: {p}", file=sys.stderr)
            sys.exit(1)
        # If it's an image, run single-image inference and optionally save, then exit
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
            img = cv2.imread(str(p))
            if img is None:
                print(f"Failed to read image: {p}", file=sys.stderr)
                sys.exit(1)
            if args.grayscale:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            results = model(img, verbose=False, device=args.device)
            out_img = draw_detections(img, results, conf_threshold=args.conf)
            cv2.imshow('EPI Hardhat Detector (image)', out_img)
            if args.save:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), out_img)
                print(f"Saved: {out_path}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        # Else try to open as video stream
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"Could not open video: {p}", file=sys.stderr)
            sys.exit(1)

    # Set a reasonable resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps = 0.0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference (Ultralytics handles resizing)
        if args.grayscale:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_proc = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        else:
            frame_proc = frame
        results = model(frame_proc, verbose=False, device=args.device)

        # Draw
        frame = draw_detections(frame, results, conf_threshold=args.conf)

        # FPS overlay
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        title = 'EPI Hardhat Detector'
        cv2.imshow(title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
# ...existing code...


if __name__ == '__main__':
    main()
