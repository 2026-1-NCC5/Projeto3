

# FECAP - Fundação de Comércio Álvares Penteado

<p align="center">
<a href= "https://www.fecap.br/"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhZPrRa89Kma0ZZogxm0pi-tCn_TLKeHGVxywp-LXAFGR3B1DPouAJYHgKZGV0XTEf4AE&usqp=CAU" alt="FECAP - Fundação de Comércio Álvares Penteado" border="0"></a>
</p>

# Aether AI 

## Integrantes: Bruno Da Silva Ribeiro 24025958, Kauan Rocha Dias 24026492 , Gabriel Henrique Coelho Marussi 24026609, Arthur Rodrigues Ferreira 24026567

## Professores Orientadores: Rafael Diogo Rossetti, Rodnil da Silva Moreira Lisboa, Rodrigo da Rosa, Victor Bruno Alexander Rosetti de Queiroz, Marcos Minoru Nakatsugawa

## Descrição

O **Aether AI** é um sistema inteligente de monitoramento e arrecadação de alimentos que utiliza Visão Computacional (YOLOv8) para automatizar o inventário de doações. O sistema é capaz de identificar itens como arroz, feijão, macarrão, óleo e fubá, estimando o peso total com base na área ocupada pelos objetos na imagem.

### Principais Funcionalidades:
- **Detecção em Tempo Real**: Identificação automática via webcam.
- **Estimativa de Peso Inteligente**: Diferencia pacotes de tamanhos variados (ex: Arroz 1kg vs 5kg).
- **Dashboard de Impacto**: Gráficos e estatísticas de arrecadação em tempo real.
- **Gestão Fiscal (CRUD)**: Interface completa para edição, exclusão e adição manual de registros.
- **Sistema de Usuários**: Login seguro e personalização de perfil.

## 🛠 Estrutura de pastas

-Raiz<br>
|<br>
|-->documentos<br>
|-->executáveis<br>
|-->imagens<br>
|-->src<br>
  &emsp;|-->Back-end<br>
  &emsp;|-->Front-end<br>
|README.md<br>

## 🛠 Instalação

<b>Servidor Web (Python/Flask):</b>

1. Certifique-se de ter o Python instalado.
2. Navegue até a pasta do código fonte:
   ```bash
   cd src/Front-end
   ```
3. Instale as dependências necessárias:
   ```bash
   pip install flask opencv-python ultralytics
   ```
4. Inicie o servidor:
   ```bash
   python app.py
   ```
5. Acesse no navegador: `http://localhost:5000`

## 💻 Configuração para Desenvolvimento

Para rodar este projeto em ambiente de desenvolvimento, você precisará:

1. **Modelo YOLO**: O arquivo `best.pt` deve estar em `src/Front-end/detector/runs/modelo/weights/best.pt`.
2. **Banco de Dados**: O sistema utiliza SQLite, criado automaticamente na primeira execução em `src/Back-end/food_inventory.db`.

**Bibliotecas principais:**
```bash
pip install ultralytics  # IA / YOLO
pip install flask        # Servidor Web
pip install opencv-python # Processamento de Imagem
```

## 📋 Licença/License
Utilize o link <https://chooser-beta.creativecommons.org/> para fazer uma licença CC BY 4.0.

## 🎓 Referências

Aqui estão as referências usadas no projeto.

1. <https://liderancasempaticas.com/>
2. <https://repositorio.ipea.gov.br/items/4b01fa99-33a3-47a3-bbf1-1c6ed01ee22f> 
3. <https://altabooks.com.br/produto/maos-a-obra-aprendizado-de-maquina-com-scikit-learn-keras-tensorflow/> 
4. <https://rafaelizbicki.com/ame/>
5. <https://revista.sodebras.com.br/index.php/revista/article/download/76/45>
6. <https://periodicos.newsciencepubl.com/arace/article/download/1339/1902/5258>
7. <https://www.fao.org/publications/sofi/2024/>
8. <https://www.ibge.gov.br/estatisticas/sociais/saude/9127-pesquisa-nacional-por-amostra-de-domicilios.html>
9. Músicas por: <a href="https://freesound.org/people/DaveJf/sounds/616544/"> DaveJf </a> e <a href="https://freesound.org/people/DRFX/sounds/338986/"> DRFX </a> ambas com Licença CC 0.  


