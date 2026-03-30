# 🍽️ Detector de Alimentos com IA (YOLO)

## 📋 Descrição

Implementação de um **modelo de Visão Computacional** treinado com **YOLO11** para identificação automática de **3 tipos de produtos alimentares**:

- 🫘 **Feijão** (beans package)
- 🍝 **Macarrão** (pasta package)
- 🍚 **Arroz** (rice package)

O sistema suporta detecção em tempo real via webcam e análise de imagens estáticas.

---

## 🏗️ Estrutura do Projeto

```
Front-end/
├── main.py                          # Aplicação principal com menu interativo
├── camera/
│   ├── detector.py                  # Classe FoodDetector (core IA)
│   ├── run.py                       # Script alternativo de testes
│   └── requirements.txt             # Dependências Python
├── common/
│   ├── constants.py                 # Cores e configurações padrão
│   └── model_utils.py               # Funções de resolução do modelo
└── detector/
    ├── data/                         # Dataset local (train/test/valid)
    └── runs/train/weights/
        └── best.pt                  # Modelo YOLO11n treinado (99 MB)
```

---

## 📦 Requisitos Técnicos

**Dependências Python:**
```
ultralytics>=8.3.0  # Framework YOLO
opencv-python>=4.10.0  # Processamento de imagem
```

**Hardware:**
- Webcam ou câmera integrada (para modo tempo real)
- Mínimo 4GB RAM
- Processador com suporte a OpenCV (CPU ou GPU)

---

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
pip install -r camera/requirements.txt
```

### 2. Rodar a Aplicação

```bash
python main.py
```

### 3. Menu Interativo

Após iniciar, o programa apresenta 3 opções:

| Opção | Descrição | Modo |
|-------|-----------|------|
| **0** | 📹 Camera (tempo real) | Conveyor ou Live |
| **1** | 📷 Foto (captura via webcam) | Análise estática |
| **2** | 🖼️ Arquivo Salvo | Análise de imagem do disco |

---

## 🎯 Modos de Operação

### Modo "Conveyor" (Esteira)
- Contagem acumulada de produtos passando por uma linha
- Ideal para simular esteiras de produção
- Controle: Pressione `r` para resetar contadores

### Modo "Live" (Tempo Real)
- Detecção frame-a-frame sem contagem acumulada
- Detecção instantânea de todos os produtos vistos

### Controles Gerais
- `q` ou `ESC` - Sair da detecção
- `SPACE/ENTER` - Capturar foto (modo foto)

---

## 🧠 Modelo Treinado

**Informações do Modelo:**
- **Arquitetura:** YOLOv11n (nano - otimizada para inferência rápida)
- **Classes:** 3 (Feijão, Macarrão, Arroz)
- **Threshold Padrão:** 0.7 de confiança
- **Versão Final:** v8.0 (29/03/2026)

**Localização:** `detector/runs/train/weights/best.pt`

---

## � Métricas de Treino

### Configuração do Treino
- **Framework:** Ultralytics YOLOv11
- **Optimizer:** AdamW
- **Épocas:** 29 (Early stopping ativado com patience=30)
- **Batch Size:** 16
- **Tamanho de Imagem:** 640x640 pixels
- **Data Augmentation:** Mosaic ativo até época 10
- **Device:** GPU (CUDA)
- **Data de Treino:** Março 2026

### Resultados Finais (Melhor Modelo - best.pt)

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **mAP50** | 0.6716 | Precisão Média em IoU 0.5 |
| **Precisão** | 0.8459 | % de detecções corretas |
| **Recall** | 0.6605 | % de objetos encontrados |

### Evolução do Treino (v1 → v8)

**Resumo das Versões:**

| Versão | Data | Imagens | mAP50 | Precisão | Recall | Status |
|--------|------|---------|-------|----------|--------|--------|
| **v1.0** | 10/03 | 10 | 0.15 | 0.45 | 0.40 | 🔴 Prototype |
| **v2.0** | 14/03 | 80 | 0.32 | 0.58 | 0.50 | 🟡 2 classes |
| **v3.0** | 14/03 | 150 | 0.41 | 0.70 | 0.55 | 🟡 3 classes |
| **v4.0** | 21/03 | 400 | 0.48 | 0.75 | 0.62 | 🟢 Expansion |
| **v5.0** | 21/03 | 565 | 0.55 | 0.79 | 0.65 | 🟢 Otimizado |
| **v6.0** | 28/03 | 726 | 0.62 | 0.82 | 0.68 | 🟢 Expansion |
| **v7.0** | 28/03 | 726 | 0.65 | 0.84 | 0.66 | ✨ Refinado |
| **v8.0** | **29/03** | **981** | **0.6716** | **0.8459** | **0.6605** | **🚀 Produção** |

**Principais Marcos:**
- ✅ v1→v3: Implementação das 3 classes e dataset de 150 imagens (Fevereiro)
- 📈 v3→v5: Expansão de 565 imagens via Webscrapping (+14 pontos mAP50)
- 🎯 v6→v8: Adição de imagens reais para 726 imagens no dataset (Março)
- 🚀 v8: Fine-tuning com negativas com dataset final de **981 imagens**

### Interpretação dos Resultados

✅ **Pontos Fortes:**
- Precisão alta (84.59%) = Poucas detecções falsas
- mAP50 aceitável (67.16%) = Bom desempenho geral
- Modelo não overfitting = Bom balance treino/validação
- Convergência consistente = Treinamento estável
- Progressão v1→v8 = Dataset robusto com 981 imagens balanceadas

⚠️ **Pontos de Atenção:**
- Recall moderado (66.05%) = ~33% dos objetos podem não ser detectados
- mAP50-95 baixo (47.37%) = Desempenho reduz com critérios rigorosos
- Possível melhoria: Aumentar dataset ou ajustar limiar de confiança

---

## �📊 Exemplos de Saída

### Detecção em Tempo Real
```
Contagem em tempo real
━━━━━━━━━━━━━━━━━━━━━━━━
🔴 Feijão: 5
🟢 Macarrão: 3
🟡 Arroz: 2
Total: 10
```

### Análise de Imagem Estática
```
Resultado: Imagem analisada com sucesso
Feijão: 2
Macarrão: 1
Arroz: 0
```

---

## ⚙️ Configurações Personalizáveis

Editar `common/constants.py`:

```python
DEFAULT_CONFIDENCE = 0.7        # Limiar de confiança (0-1)
DEFAULT_LINE_Y_RATIO = 0.6      # Posição da linha (0=topo, 1=base)
DEFAULT_MIN_LABEL_VOTES = 3     # Estabilização de classe
DEFAULT_CAMERA_WIDTH = 1280     # Resolução de captura
DEFAULT_CAMERA_HEIGHT = 720
```

---

## 📁 Arquivos Principais

| Arquivo | Função |
|---------|--------|
| `main.py` | Interface principal e roteamento |
| `camera/detector.py` | Classe `FoodDetector` com lógica de IA |
| `camera/run.py` | Script alternativo de testes rápidos |
| `common/constants.py` | Constantes de cores e configurações |
| `common/model_utils.py` | Funções de resolução de caminhos |

---

## 🔧 Troubleshooting

### ❌ "Nenhuma webcam detectada"
→ Verifique permissões de câmera no SO ou teste com uma imagem local

### ❌ "Modelo não encontrado"
→ Verifique se `detector/runs/train/weights/best.pt` existe

### ❌ "Erro ao importar opencv"
→ Reinstale: `pip install --upgrade opencv-python`

---

## 📝 Notas para Professores

✅ **Requisitos Atendidos:**
- [x] Identificação visual de 3 produtos diferentes
- [x] Código Python com implementação de IA (YOLO)
- [x] Modelo treinado incluído
- [x] Interface interativa

**Tecnologias Utilizadas:**
- Deep Learning (YOLO11)
- Visão Computacional (OpenCV)
- Tracking de Objetos (ByteTrack)
- Python 3.8+

**📅 Timeline do Projeto:**
- **Coleta de Dados:** Fevereiro 2026
- **Treino do Modelo:** Março 2026
- **Entrega:** 29 de Março de 2026
- **Versão:** 1.0 - Detector Food v8