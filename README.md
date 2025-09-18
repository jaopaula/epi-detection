

# EPI Detection

Detecção de Equipamentos de Proteção Individual (EPI) em tempo real usando YOLOv8 e Python. Este projeto detecta capacetes, óculos e outros EPIs em imagens, vídeos ou webcam, com suporte para treinamento personalizado.

## Funcionalidades
- Detecção em tempo real via webcam, imagem ou vídeo
- Treinamento e ajuste fino com seu próprio conjunto de dados
- Estrutura pronta para adicionar novos tipos de EPI

## Como usar

### 1. Clonar o repositório
```bash
git clone https://github.com/jaopaula/epi-detection.git
cd epi-detection
```

### 2. Instalar dependências
Recomenda-se usar um ambiente virtual:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```


O projeto já possui a estrutura de pastas em `data/`, mas está vazia. Siga os passos abaixo para baixar e preparar os dados:

#### 3.1 Baixar o dataset de EPI
- Dataset utilizado: [Hard Hat Detection - Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- Faça login no Kaggle e baixe o arquivo ZIP do dataset.
- Extraia o conteúdo em uma pasta de sua preferência.

#### 3.2 Converter o dataset para YOLO
- O dataset vem em formato COCO (JSON) e VOC (XML). Para usar com YOLOv8, converta usando o script do projeto:

```bash
python scripts/coco2yolo.py
```
- Antes de rodar, ajuste os caminhos no início do script `coco2yolo.py` para apontar para o local onde você extraiu o dataset.
- O script irá criar as pastas `data/images/train`, `data/images/val`, `data/labels/train`, `data/labels/val` e gerar o arquivo `data/dataset.yaml`.

#### 3.3 Conferir estrutura final
- Após a conversão, as imagens e labels estarão organizadas nas pastas corretas:
  - `data/images/train/` e `data/images/val/`
  - `data/labels/train/` e `data/labels/val/`
- O arquivo `data/dataset.yaml` estará pronto para uso.

### 4. Treinar o modelo
```bash
python scripts/train.py --data data/dataset.yaml --epochs 50 --batch 16 --img 640 --weights yolov8n.pt --project runs/train --name hardhat-color-invariant
```
O modelo treinado será salvo em `runs/train/hardhat-color-invariant/weights/best.pt`.

### 5. Rodar a detecção
Para usar a webcam:
```bash
python app.py --weights runs/train/hardhat-color-invariant/weights/best.pt --source 0
```
Para testar com uma imagem:
```bash
python app.py --weights runs/train/hardhat-color-invariant/weights/best.pt --source caminho/da/imagem.jpg
```

## Estrutura do projeto
```
├── app.py                # Detecção em tempo real
├── scripts/train.py      # Treinamento do modelo
├── data/                 # Estrutura de dados (imagens/labels)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── requirements.txt      # Dependências
├── .gitignore            # Arquivos/pastas ignorados
└── README.md             # Este guia
```

## Observações
- O conjunto de dados e os modelos treinados não são enviados ao GitHub para manter o repositório leve.
- Siga o README para preparar os dados e treinar o modelo.
- Para adicionar novos tipos de EPI, basta ajustar o conjunto de dados e o arquivo `dataset.yaml`.

## Contribuição
Pull requests são bem-vindos! Sinta-se à vontade para abrir issues ou sugerir melhorias.

## Licença
Este projeto é open-source sob a licença MIT.

Um exemplo de `dataset.yaml` é fornecido e assume uma única classe: `hardhat` (capacete).


## Observações técnicas
- Requer uma webcam funcional para detecção em tempo real.
- Testado com Python 3.10/3.11.
- Se preferir rodar apenas na CPU, o sistema funcionará, porém será mais lento.
- Para melhores resultados, faça o ajuste fino (fine-tuning) do modelo para a classe `hardhat` e forneça o caminho do modelo treinado via `--weights caminho/para/hardhat.pt`.

### Solução de problemas
- Se a instalação falhar no NumPy usando Python 3.13, prefira Python 3.11 (`py -3.11`) ou instale o Microsoft C++ Build Tools e garanta que uma versão compatível do NumPy esteja disponível.
