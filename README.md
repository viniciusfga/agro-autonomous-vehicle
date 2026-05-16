# 🌱 Agro Autonomous Vehicle - OpenCV Branch

Sistema de navegação agrícola baseado em Visão Computacional utilizando OpenCV, Raspberry Pi e Arduino.

Esta branch contém os códigos responsáveis pela coleta de imagens, processamento visual e detecção de linhas de plantio para navegação autônoma em ambiente agrícola.

---

## 📌 Objetivo

Desenvolver um sistema de baixo custo para agricultura de precisão capaz de:

- Detectar linhas de cultivo
- Navegar entre leiras agrícolas
- Coletar imagens para datasets
- Processar visão computacional em tempo real
- Integrar sensores e atuadores embarcados

---

## 🧠 Tecnologias Utilizadas

- Python
- OpenCV
- NumPy
- Raspberry Pi
- Arduino
- Câmera USB / CSI
- Serial Communication

---

## ⚙️ Funcionalidades

- Captura de imagens e vídeos
- Pré-processamento de imagens
- Detecção de linhas de cultivo
- Segmentação por cor
- Aplicação de filtros
- Conversão de datasets
- Comunicação serial com Arduino
- Testes de navegação

---

## 📂 Estrutura do Projeto

```bash
agro-autonomous-vehicle/
│
├── capturas/                 # Imagens coletadas
├── datasets/                 # Bases de dados
├── labels/                   # Anotações
│
├── coletar_dataset.py        # Captura de imagens
├── detectar_linhas.py        # Navegação por OpenCV
├── processar_dataset.py      # Processamento de imagens
│
├── arduino/                  # Códigos embarcados
├── docs/                     # Documentação
└── README.md
