
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Converte COCO para YOLO para múltiplos EPIs")
    parser.add_argument('--epi', type=str, nargs='+', required=True, help="Lista de EPIs (ex: helmet glasses)")
    parser.add_argument('--split', type=float, default=0.85, help="Proporção para treino")
    parser.add_argument('--output', type=str, default='data', help="Pasta destino para YOLO")
    return parser.parse_args()

args = parse_args()
COCO_ANNOTATION_PATHS = [f'data/original/{epi}/annotations/train.json' for epi in args.epi]
IMAGES_DIRS = [f'data/original/{epi}/images' for epi in args.epi]
OUTPUT_DIR = args.output
TRAIN_SPLIT = args.split

COCO_CLASSES = ["head", "helmet", "person", "vest", "glasses"]
YOLO_CLASSES = args.epi  # Lista de EPIs
COCO_TO_YOLO = {COCO_CLASSES.index(epi): idx for idx, epi in enumerate(YOLO_CLASSES) if epi in COCO_CLASSES}


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

    all_images = {}
    all_annos = []
    # Carrega todos os COCOs
    for coco_path in COCO_ANNOTATION_PATHS:
        if Path(coco_path).exists():
            with open(coco_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
            for img in coco["images"]:
                all_images[img["id"]] = img
            all_annos.extend(coco["annotations"])
        else:
            print(f"Arquivo {coco_path} não encontrado.")

    # Split train/val
    img_ids = list(all_images.keys())
    split_idx = int(len(img_ids) * TRAIN_SPLIT)
    train_ids = set(img_ids[:split_idx])
    val_ids = set(img_ids[split_idx:])

    # Agrupa anotações por imagem
    img_to_annos = {}
    for anno in all_annos:
        img_id = anno["image_id"]
        if img_id not in img_to_annos:
            img_to_annos[img_id] = []
        img_to_annos[img_id].append(anno)

    # Processa imagens
    for img_id, img_info in tqdm(all_images.items(), desc="Convertendo imagens"):
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]
        src_img = None
        for img_dir in IMAGES_DIRS:
            candidate = Path(img_dir) / file_name
            if candidate.exists():
                src_img = candidate
                break
        if src_img is None:
            print(f"Imagem {file_name} não encontrada em nenhuma pasta de EPIs.")
            continue
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
            # Descobre nome da classe
            if cat_id < len(COCO_CLASSES):
                cls_name = COCO_CLASSES[cat_id]
            else:
                continue
            if cls_name not in YOLO_CLASSES:
                continue  # Ignora classes não desejadas
            yolo_cls = YOLO_CLASSES.index(cls_name)
            bbox = convert_bbox_coco_to_yolo(anno["bbox"], img_w, img_h)
            line = f"{yolo_cls} {' '.join(f'{v:.6f}' for v in bbox)}"
            lines.append(line)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # Gera dataset.yaml multi-classe
    yaml_path = Path(OUTPUT_DIR) / f"dataset_{'_'.join(YOLO_CLASSES)}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""path: ../data\ntrain: images/train\nval: images/val\nnc: {len(YOLO_CLASSES)}\nnames: {YOLO_CLASSES}\n""")
    print(f"\nConversão concluída! Arquivo dataset.yaml gerado em {yaml_path}")

if __name__ == "__main__":
    main()
