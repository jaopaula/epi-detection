import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm


# CONFIGURAÇÃO: ATENÇÃO!
# Antes de rodar este script, ajuste os caminhos abaixo conforme o local onde você baixou o dataset do Kaggle.
# Exemplo de download: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
#
# COCO_ANNOTATION_PATH: caminho para o arquivo train.json das anotações
# IMAGES_DIR: pasta onde estão as imagens originais
# OUTPUT_DIR: pasta destino para o dataset convertido (recomenda-se manter 'data')
#
# Exemplo:
# COCO_ANNOTATION_PATH = r'C:/Users/seu_usuario/.cache/kagglehub/datasets/andrewmvd/hard-hat-detection/versions/1/annotations/train.json'
# IMAGES_DIR = r'C:/Users/seu_usuario/.cache/kagglehub/datasets/andrewmvd/hard-hat-detection/versions/1/images'

COCO_ANNOTATION_PATH = r'C:/Users/evosystem04.ti/.cache/kagglehub/datasets/andrewmvd/hard-hat-detection/versions/1/annotations/train.json'  # Caminho para o arquivo COCO
IMAGES_DIR = r'C:/Users/evosystem04.ti/.cache/kagglehub/datasets/andrewmvd/hard-hat-detection/versions/1/images'  # Pasta com imagens originais
OUTPUT_DIR = "data"  # Pasta destino para YOLO (relativo à raiz do projeto)
TRAIN_SPLIT = 0.85  # Proporção para treino

# Classes do dataset Kaggle
COCO_CLASSES = ["head", "helmet", "person", "vest"]
YOLO_CLASSES = ["helmet"]  # Use só capacete ou todas se quiser

# Mapeamento COCO -> YOLO (apenas capacete)
COCO_TO_YOLO = {1: 0}  # COCO id 1 = helmet, YOLO id 0


def convert_bbox_coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return [x_center, y_center, w, h]


def main():
    # Cria pastas
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        Path(f"{OUTPUT_DIR}/{sub}").mkdir(parents=True, exist_ok=True)

    # Carrega COCO
    with open(COCO_ANNOTATION_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annos = coco["annotations"]

    # Split train/val
    img_ids = list(images.keys())
    split_idx = int(len(img_ids) * TRAIN_SPLIT)
    train_ids = set(img_ids[:split_idx])
    val_ids = set(img_ids[split_idx:])

    # Agrupa anotações por imagem
    img_to_annos = {}
    for anno in annos:
        img_id = anno["image_id"]
        if img_id not in img_to_annos:
            img_to_annos[img_id] = []
        img_to_annos[img_id].append(anno)

    # Processa imagens
    for img_id, img_info in tqdm(images.items(), desc="Convertendo imagens"):
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]
        src_img = Path(IMAGES_DIR) / file_name
        if img_id in train_ids:
            dst_img = Path(OUTPUT_DIR) / "images/train" / file_name
            dst_lbl = Path(OUTPUT_DIR) / "labels/train" / (file_name.replace(".jpg", ".txt"))
        else:
            dst_img = Path(OUTPUT_DIR) / "images/val" / file_name
            dst_lbl = Path(OUTPUT_DIR) / "labels/val" / (file_name.replace(".jpg", ".txt"))
        shutil.copy2(src_img, dst_img)

        # Cria label YOLO
        lines = []
        for anno in img_to_annos.get(img_id, []):
            cat_id = anno["category_id"]
            if cat_id not in COCO_TO_YOLO:
                continue  # Ignora classes não desejadas
            yolo_cls = COCO_TO_YOLO[cat_id]
            bbox = convert_bbox_coco_to_yolo(anno["bbox"], img_w, img_h)
            line = f"{yolo_cls} {' '.join(f'{v:.6f}' for v in bbox)}"
            lines.append(line)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # Gera dataset.yaml
    yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""path: ../data\ntrain: images/train\nval: images/val\nnc: 1\nnames: [helmet]\n""")
    print(f"\nConversão concluída! Arquivo dataset.yaml gerado em {yaml_path}")

if __name__ == "__main__":
    main()
