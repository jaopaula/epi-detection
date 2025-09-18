import subprocess
import sys
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepara o dataset para o EPI desejado")
    parser.add_argument('--epi', type=str, nargs='+', required=True, help="Lista de EPIs (ex: helmet glasses)")
    return parser.parse_args()

args = parse_args()

epi_args = args.epi
print(f"Convertendo COCO para YOLO para os EPIs: {epi_args}")
c2y_path = Path("scripts/coco2yolo.py")
if not c2y_path.exists():
    print("Erro: scripts/coco2yolo.py não encontrado.")
    sys.exit(1)

# Executa conversão usando lista de EPIs
subprocess.run([sys.executable, str(c2y_path), '--epi', *epi_args], check=True)
yaml_name = f"dataset_{'_'.join(epi_args)}.yaml"
print("\nDataset pronto para treino!")
print(f"Use: python scripts/train.py --data data/{yaml_name} --epochs 50 --batch 16 --img 640 --weights yolov8n.pt")
