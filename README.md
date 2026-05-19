# 🤖 Agro Autonomous Vehicle — YOLO Branch

Branch dedicada ao desenvolvimento de navegação agrícola autônoma utilizando Deep Learning, YOLOv8 e Visão Computacional.

O objetivo desta implementação é substituir abordagens tradicionais baseadas apenas em OpenCV por modelos de detecção mais robustos para operação em ambiente agrícola real.

---

# 🌱 Sobre o Projeto

Este projeto faz parte do desenvolvimento de um veículo autônomo agrícola de baixo custo voltado para Agricultura 4.0 e agricultura familiar.

A proposta utiliza:

- Visão Computacional
- Redes Neurais Convolucionais (CNN)
- Detecção de linhas de cultivo
- Navegação autônoma
- Inferência em tempo real

---

# 🎯 Objetivos da Branch

- Implementar navegação agrícola com YOLOv8
- Detectar linhas de plantio
- Realizar inferência em tempo real
- Validar modelos leves embarcados
- Integrar Raspberry Pi / Jetson / Arduino
- Evoluir para navegação autônoma inteligente

---

# 🧠 Tecnologias

| Tecnologia | Finalidade |
|---|---|
| Python | Desenvolvimento principal |
| YOLOv8 | Detecção de objetos |
| Ultralytics | Framework YOLO |
| OpenCV | Processamento visual |
| PyTorch | Deep Learning |
| NumPy | Manipulação numérica |
| Raspberry Pi | Computação embarcada |
| Arduino | Controle de atuadores |

---

# 🚜 Arquitetura do Sistema

```text
Câmera
   ↓
YOLOv8
   ↓
Detecção de linhas de cultivo
   ↓
Processamento da navegação
   ↓
Decisão de direção
   ↓
Arduino
   ↓
Motores
```

---

# 📂 Estrutura do Projeto

```bash
agro-autonomous-vehicle/

├── datasets/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│
├── runs/
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── webcam.py
│   └── convert_annotations.py
│
├── weights/
│
├── dataset.yaml
├── requirements.txt
└── README.md
```

---

# 📸 Dataset

O dataset é focado em:

- Crop Row Detection
- Linhas agrícolas
- Navegação entre leiras
- Ambientes agrícolas reais
- Diferentes iluminações
- Diferentes resoluções

Formato utilizado:

- YOLO TXT
- Imagens RGB
- Bounding Boxes

---

# 🏋️ Treinamento

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Instalar YOLOv8

```bash
pip install ultralytics
```

## Treinar modelo

```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=640
```

---

# 🔍 Inferência

Imagem:

```bash
yolo task=detect mode=predict model=best.pt source=imagem.jpg
```

Webcam:

```bash
python scripts/webcam.py
```

Vídeo:

```bash
yolo task=detect mode=predict model=best.pt source=video.mp4
```

---

# 📊 Modelos Planejados

- [x] YOLOv8n
- [ ] YOLOv8s
- [ ] YOLOv9
- [ ] RT-DETR
- [ ] CNN customizada
- [ ] Implementação baseada em SN-CNN

---

# 📚 Base Científica

O desenvolvimento desta branch é inspirado em pesquisas recentes sobre:

- Agricultura Inteligente
- Navegação agrícola autônoma
- Deep Learning aplicado ao agro
- Extração de linhas de cultivo
- CNNs leves para sistemas embarcados

Artigos relacionados:

- SN-CNN
- YOLO aplicado à agricultura
- Crop Row Detection
- Agricultural Robotics

---
