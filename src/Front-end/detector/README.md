# Detector - Dados e Resultados

Este diretório guarda os arquivos de treino e os artefatos gerados pelo treinamento do modelo.

## Estrutura

- `data/`: dataset local usado no treino/validação/teste (pastas `train/`, `valid/` e `test/`).
- `runs/`: saídas dos treinamentos, incluindo métricas, gráficos, logs e pesos finais (`best.pt`).

## Datas (referência local)

- `data/`: atualizado em 29/03/2026 18:55
- `runs/`: atualizado em 29/03/2026 20:21

## Uso rápido

- Para apenas executar a aplicação: o arquivo essencial é `runs/train/weights/best.pt`.
- Para novo treino ou ajuste: usar também os dados em `data/`.