import subprocess
import sys
from pathlib import Path

print("Convertendo COCO para YOLO usando diretórios do projeto...")
c2y_path = Path("scripts/coco2yolo.py")
if not c2y_path.exists():
    print("Erro: scripts/coco2yolo.py não encontrado.")
    sys.exit(1)

# Executa conversão usando os diretórios já configurados no coco2yolo.py
subprocess.run([sys.executable, str(c2y_path)], check=True)
print("\nDataset pronto para treino!")
print("Use: python scripts/train.py --data data/dataset.yaml --epochs 50 --batch 16 --img 640 --weights yolov8n.pt")
