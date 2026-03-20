import cv2
import numpy as np
import pandas as pd

# Carregar imagem (OpenCV lê em BGR)
img = cv2.imread("arroz-branco-camil-t2-1kg-300x300.jpg")

# Converter de BGR para RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mostrar dimensões
altura, largura, canais = img_rgb.shape
print(f"Dimensões da imagem: {altura} x {largura} x {canais}")

# Transformar em lista de pixels
pixels = img_rgb.reshape(-1, 3)

# Criar DataFrame
df = pd.DataFrame(pixels, columns=["R", "G", "B"])

# Salvar CSV
df.to_csv("matriz_arroz.csv", index=False)

print("CSV gerado com sucesso!")