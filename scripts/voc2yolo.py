import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm

# CONFIG
VOC_ANNOTATIONS_DIR = 'data/original/annotations'  # Pasta de anotações VOC (XML)
IMAGES_DIR = 'data/original/images'                # Pasta de imagens originais
OUTPUT_DIR = 'data'                                # Pasta destino para YOLO
TRAIN_SPLIT = 0.85

# Classes do dataset
CLASSES = ['helmet']  # Adicione outras se quiser

# Cria pastas destino
for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    Path(f'{OUTPUT_DIR}/{sub}').mkdir(parents=True, exist_ok=True)

# Lista todos XMLs
xml_files = list(Path(VOC_ANNOTATIONS_DIR).glob('*.xml'))

# Split train/val
split_idx = int(len(xml_files) * TRAIN_SPLIT)
train_xmls = xml_files[:split_idx]
val_xmls = xml_files[split_idx:]


def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def process_xml(xml_path, img_dst, lbl_dst):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_file = root.find('filename').text
    img_src = Path(IMAGES_DIR) / img_file
    shutil.copy2(img_src, img_dst)
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls != 'helmet':
            continue
        xmlbox = obj.find('bndbox')
        b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)]
        bbox = convert_bbox((w, h), b)
        lines.append(f"0 {' '.join(f'{v:.6f}' for v in bbox)}")
    with open(lbl_dst, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# Processa treino
for xml_path in tqdm(train_xmls, desc='Processando treino'):
    img_name = xml_path.stem + '.jpg'
    img_dst = Path(OUTPUT_DIR) / 'images/train' / img_name
    lbl_dst = Path(OUTPUT_DIR) / 'labels/train' / (img_name.replace('.jpg', '.txt'))
    process_xml(xml_path, img_dst, lbl_dst)

# Processa validação
for xml_path in tqdm(val_xmls, desc='Processando validação'):
    img_name = xml_path.stem + '.jpg'
    img_dst = Path(OUTPUT_DIR) / 'images/val' / img_name
    lbl_dst = Path(OUTPUT_DIR) / 'labels/val' / (img_name.replace('.jpg', '.txt'))
    process_xml(xml_path, img_dst, lbl_dst)

# Gera dataset.yaml
yaml_path = Path(OUTPUT_DIR) / 'dataset.yaml'
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"""path: ../data\ntrain: images/train\nval: images/val\nnc: 1\nnames: [helmet]\n""")
print(f"\nConversão VOC->YOLO concluída! Arquivo dataset.yaml gerado em {yaml_path}")
