import kagglehub
import os
import sys
import subprocess
from pathlib import Path

# 1. Baixar o dataset do Kaggle
print("Baixando dataset do Kaggle...")
dataset_path = kagglehub.dataset_download("andrewmvd/hard-hat-detection")
print(f"Dataset baixado em: {dataset_path}")

# 2. Executar conversão COCO -> YOLO
# Ajusta caminhos para o script coco2yolo.py
coco_ann_path = Path(dataset_path) / "annotations/train.json"
images_dir = Path(dataset_path) / "images"

# Edita coco2yolo.py para usar os caminhos corretos
c2y_path = Path("scripts/coco2yolo.py")
if not c2y_path.exists():
    print("Erro: scripts/coco2yolo.py não encontrado.")
    sys.exit(1)

# Atualiza variáveis do script coco2yolo.py
with open(c2y_path, "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("COCO_ANNOTATION_PATH = \"../hard-hat-detection/annotations/train.json\"", f"COCO_ANNOTATION_PATH = r'{coco_ann_path}'")
code = code.replace("IMAGES_DIR = \"../hard-hat-detection/images/\"", f"IMAGES_DIR = r'{images_dir}'")
with open(c2y_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Convertendo COCO para YOLO...")
subprocess.run([sys.executable, str(c2y_path)], check=True)
print("\nDataset pronto para treino!")
print("Use: python scripts/train.py --data data/dataset.yaml --epochs 50 --batch 16 --img 640 --weights yolov8n.pt")
