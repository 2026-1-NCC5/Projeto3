import csv
import os
from collections import Counter
from datetime import datetime

import cv2
from ultralytics import YOLO


MODELO_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
EQUIPE_ATIVA = "Equipe_A"
ITEM_ALVO = None  # Exemplo: "bottle" para filtrar apenas uma classe.
CONFIANCA_MINIMA = 0.40
DISTANCIA_MAXIMA_TRACK = 70
MAX_FRAMES_SEM_ATUALIZACAO = 20
PASTA_EVIDENCIAS = "evidencias"
ARQUIVO_EVENTOS = "eventos_contagem.csv"


def distancia(p1, p2):
	dx = p1[0] - p2[0]
	dy = p1[1] - p2[1]
	return (dx * dx + dy * dy) ** 0.5


def garantir_arquivo_eventos(caminho_csv):
	if os.path.exists(caminho_csv):
		return

	with open(caminho_csv, "w", newline="", encoding="utf-8") as arquivo:
		writer = csv.writer(arquivo)
		writer.writerow(
			[
				"timestamp",
				"equipe",
				"categoria",
				"confianca",
				"track_id",
				"centro_x",
				"centro_y",
				"evidencia",
			]
		)


def salvar_evento(caminho_csv, equipe, categoria, confianca, track_id, centro, evidencia):
	with open(caminho_csv, "a", newline="", encoding="utf-8") as arquivo:
		writer = csv.writer(arquivo)
		writer.writerow(
			[
				datetime.now().isoformat(timespec="seconds"),
				equipe,
				categoria,
				f"{confianca:.4f}",
				track_id,
				centro[0],
				centro[1],
				evidencia,
			]
		)


def main():
	os.makedirs(PASTA_EVIDENCIAS, exist_ok=True)
	garantir_arquivo_eventos(ARQUIVO_EVENTOS)

	modelo = YOLO(MODELO_PATH)
	camera = cv2.VideoCapture(CAMERA_INDEX)

	if not camera.isOpened():
		print("Erro ao iniciar a camera.")
		return

	print("Iniciando a camera... Pressione Q para sair")

	contagem_total = Counter()
	tracks = {}
	proximo_track_id = 1

	while True:
		sucesso, frame = camera.read()
		if not sucesso:
			print("Erro ao acessar a camera.")
			break

		altura, largura = frame.shape[:2]
		linha_contagem_y = int(altura * 0.55)

		frame_anotado = frame.copy()
		resultados = modelo(frame, stream=True, conf=CONFIANCA_MINIMA)

		deteccoes = []
		for resultado in resultados:
			frame_anotado = resultado.plot()
			if resultado.boxes is None or len(resultado.boxes) == 0:
				continue

			classes_ids = resultado.boxes.cls.tolist()
			confiancas = resultado.boxes.conf.tolist()
			caixas = resultado.boxes.xyxy.tolist()
			nomes = resultado.names

			for cls_id, conf, caixa in zip(classes_ids, confiancas, caixas):
				categoria = nomes[int(cls_id)]
				if ITEM_ALVO and categoria != ITEM_ALVO:
					continue

				x1, y1, x2, y2 = caixa
				cx = int((x1 + x2) / 2)
				cy = int((y1 + y2) / 2)
				deteccoes.append(
					{
						"categoria": categoria,
						"confianca": float(conf),
						"centro": (cx, cy),
					}
				)

		for track_id in list(tracks.keys()):
			tracks[track_id]["missing"] += 1
			if tracks[track_id]["missing"] > MAX_FRAMES_SEM_ATUALIZACAO:
				del tracks[track_id]

		for deteccao in deteccoes:
			categoria = deteccao["categoria"]
			centro = deteccao["centro"]
			confianca = deteccao["confianca"]

			melhor_track = None
			menor_distancia = DISTANCIA_MAXIMA_TRACK

			for track_id, dados in tracks.items():
				if dados["categoria"] != categoria:
					continue

				dist = distancia(centro, dados["centro"])
				if dist < menor_distancia:
					menor_distancia = dist
					melhor_track = track_id

			if melhor_track is None:
				melhor_track = proximo_track_id
				proximo_track_id += 1
				tracks[melhor_track] = {
					"categoria": categoria,
					"centro": centro,
					"ultimo_y": centro[1],
					"counted": False,
					"missing": 0,
				}
			else:
				tracks[melhor_track]["centro"] = centro
				tracks[melhor_track]["missing"] = 0

			ultimo_y = tracks[melhor_track]["ultimo_y"]
			atual_y = centro[1]

			cruzou_linha = ultimo_y < linha_contagem_y <= atual_y
			if cruzou_linha and not tracks[melhor_track]["counted"]:
				contagem_total[categoria] += 1
				tracks[melhor_track]["counted"] = True

				timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
				nome_evidencia = f"{timestamp}_{categoria}_{melhor_track}.jpg"
				caminho_evidencia = os.path.join(PASTA_EVIDENCIAS, nome_evidencia)
				cv2.imwrite(caminho_evidencia, frame_anotado)

				salvar_evento(
					ARQUIVO_EVENTOS,
					EQUIPE_ATIVA,
					categoria,
					confianca,
					melhor_track,
					centro,
					caminho_evidencia,
				)

			tracks[melhor_track]["ultimo_y"] = atual_y

		cv2.line(frame_anotado, (0, linha_contagem_y), (largura, linha_contagem_y), (0, 255, 255), 2)
		cv2.putText(
			frame_anotado,
			f"Equipe: {EQUIPE_ATIVA}",
			(20, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
		)

		y_pos = 65
		cv2.rectangle(frame_anotado, (10, 40), (380, 230), (0, 0, 0), -1)
		cv2.putText(
			frame_anotado,
			"Contagem total (sem duplicidade):",
			(20, y_pos),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 255),
			2,
		)
		y_pos += 30

		for item, quantidade in sorted(contagem_total.items()):
			texto_contagem = f"{item}: {quantidade}"
			cv2.putText(
				frame_anotado,
				texto_contagem,
				(20, y_pos),
				cv2.FONT_HERSHEY_COMPLEX,
				0.7,
				(255, 255, 255),
				2,
			)
			y_pos += 28

		cv2.imshow("Contador de alimentos", frame_anotado)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	camera.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


# Como esse fluxo funciona agora:
# 1. A webcam captura um frame.
# 2. O YOLO detecta objetos no frame e retorna classe, confianca e caixa.
# 3. O sistema converte cada caixa no ponto central do objeto.
# 4. Esse ponto e associado a um track existente (ou cria um novo track).
# 5. Se o track cruza a linha de contagem e ainda nao foi contado:
#    - incrementa a categoria,
#    - salva imagem de evidencia,
#    - grava uma linha no CSV.
# 6. A interface mostra video anotado, linha de contagem e placar total por categoria.