# FECAP - Fundação de Comércio Álvares Penteado

<p align="center">
<a href= "https://www.fecap.br/"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhZPrRa89Kma0ZZogxm0pi-tCn_TLKeHGVxywp-LXAFGR3B1DPouAJYHgKZGV0XTEf4AE&usqp=CAU" alt="FECAP - Fundação de Comércio Álvares Penteado" border="0"></a>
</p>

# Aether AI 

## Integrantes: 
- Bruno Da Silva Ribeiro (24025958)
- Kauan Rocha Dias (24026492)
- Gabriel Henrique Coelho Marussi (24026609)
- Arthur Rodrigues Ferreira (24026567)

## Professores orientadores
- Rafael Diogo Rossetti
- Rodnil da Silva Moreira Lisboa
- Rodrigo da Rosa
- Victor Bruno Alexander Rosetti de Queiroz
- Marcos Minoru Nakatsugawa

## Descrição
O Aether AI automatiza a identificação e o registro de doações alimentares a partir de imagens, usando modelos YOLO (Ultralytics) para detectar itens (ex.: arroz, feijão, macarrão, óleo, fubá) e estimar peso aproximado a partir da área ocupada. O sistema inclui captura em tempo real, processamento de imagens, armazenamento em banco de dados e visualização via dashboard web.

## Funcionalidades principais
- Detecção em tempo real via webcam ou upload de imagens.
- Estimativa de peso por item com base na área detectada e regras por categoria.
- Dashboard com gráficos de arrecadação e histórico de doações.
- CRUD para gerenciar registros de doações e itens.
- Autenticação de usuários (login/registro) e gestão básica de perfis.

## 🛠 Estrutura de pastas
- Código-fonte: [src/Front-end](src/Front-end) — servidor Flask, front-end e integração com detector.
- Backend e banco: [src/Back-end](src/Back-end) — scripts para banco de dados e lógica de persistência.
- Detector (YOLO): [src/Front-end/detector](src/Front-end/detector) — modelo, captura e scripts de inferência.

## Requisitos
- Python 3.8+ (recomendado 3.10/3.11)
- pip
- Requisitos Python listados em [src/Front-end/requirements.txt](src/Front-end/requirements.txt) e [src/Front-end/camera/requirements.txt](src/Front-end/camera/requirements.txt) (se for usar a câmera).

## Instalação e execução (desenvolvimento)
1. Clone o repositório e abra o diretório do projeto:
   ```bash
   cd Projeto3
   ```
2. Recomenda-se criar e ativar um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Instale dependências principais (no diretório raiz ou em `src/Front-end`):
   ```bash
   pip install -r src/Front-end/requirements.txt
   # se for usar a câmera
   pip install -r src/Front-end/camera/requirements.txt
   ```
4. Configure o modelo YOLO (treinado):
   - Coloque o arquivo do modelo treinado `best.pt` em: [src/Front-end/detector/runs/modelo/weights/best.pt](src/Front-end/detector/runs/modelo/weights/best.pt)
5. Banco de dados:
   - O SQLite é criado automaticamente na primeira execução. Por padrão o arquivo fica em [src/Back-end](src/Back-end) (ver `database.py`).
6. Inicie a aplicação (Flask):
   ```bash
   cd src/Front-end
   python app.py
   ```
7. Abra no navegador: http://localhost:5000

### Configuração adicional
- Variáveis de ambiente (opcionais):
  - `FLASK_ENV=development` — ativa modo de desenvolvimento do Flask.
  - `MODEL_PATH` — caminho para o `.pt` se for diferente do padrão.

### Observações e dicas
- Se a aplicação não encontrar o modelo, verifique o caminho e permissões do arquivo `best.pt`.
- Para rodar apenas o detector por linha de comando, veja os scripts em [src/Front-end/detector](src/Front-end/detector).
- Logs e resultados de inferência são salvos em [src/Front-end/detector/runs](src/Front-end/detector/runs).

## Licença
Aether AI  © 2026 by Gabriel Henrique Coelho Marussi; Arthur Rodrigues Ferreira; Bruno Da Silva Ribeiro; Kauan Rocha Dias is licensed under CC BY 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/

## 🎓 Referências

Aqui estão as referências usadas no projeto.

1. <https://liderancasempaticas.com/>
2. <https://repositorio.ipea.gov.br/items/4b01fa99-33a3-47a3-bbf1-1c6ed01ee22f> 
3. <https://altabooks.com.br/produto/maos-a-obra-aprendizado-de-maquina-com-scikit-learn-keras-tensorflow/> 
4. <https://rafaelizbicki.com/ame/>
5. <https://revista.sodebras.com.br/index.php/revista/article/download/76/45>
6. <https://periodicos.newsciencepubl.com/arace/article/download/1339/1902/5258>
7. <https://www.fao.org/publications/sofi/2024/>
8. Músicas por: <a href="https://freesound.org/people/DaveJf/sounds/616544/"> DaveJf </a> e <a href="https://freesound.org/people/DRFX/sounds/338986/"> DRFX </a> ambas com Licença CC 0. 