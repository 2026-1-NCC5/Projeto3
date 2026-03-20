import cv2
import pandas as pd
import numpy as np

# Ler CSV
df = pd.read_csv("matriz_arroz.csv")

# Converter para array
pixels = df.values

# Dimensões (coloque as reais que apareceram antes)
altura = 300
largura = 300

# Reconstruir matriz
img_rgb = pixels.reshape((altura, largura, 3)).astype(np.uint8)

# Converter RGB para BGR (OpenCV precisa disso)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Salvar imagem
cv2.imwrite("saco_de_arroz_reconstruido.jpg", img_bgr)

print("Imagem reconstruída com sucesso!")