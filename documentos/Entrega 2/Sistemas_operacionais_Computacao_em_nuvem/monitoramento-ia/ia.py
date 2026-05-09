import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Ler dados
df = pd.read_csv('dados_monitorados.csv')

# -------------------------
# REGRESSÃO LINEAR
# -------------------------

df['tempo'] = np.arange(len(df))

X = df[['tempo']]
y = df['cpu']

modelo = LinearRegression()
modelo.fit(X, y)

previsao = modelo.predict([[len(df)+10]])

print('Previsão futura CPU:')
print(previsao)

# Gráfico regressão

plt.figure(figsize=(10,5))

plt.plot(df['tempo'], df['cpu'], label='CPU Real')

plt.plot(
    df['tempo'],
    modelo.predict(X),
    label='Regressão Linear'
)

plt.legend()

plt.title('Regressão Linear CPU')

plt.savefig('grafico_regressao.png')

# -------------------------
# K-MEANS
# -------------------------

X_cluster = df[['cpu', 'memoria']]

kmeans = KMeans(n_clusters=3)

kmeans.fit(X_cluster)

df['cluster'] = kmeans.labels_

plt.figure(figsize=(8,6))

plt.scatter(
    df['cpu'],
    df['memoria'],
    c=df['cluster']
)

plt.xlabel('CPU')
plt.ylabel('Memória')

plt.title('Clusters K-Means')

plt.savefig('grafico_kmeans.png')

# -------------------------
# ISOLATION FOREST
# -------------------------

iso = IsolationForest(contamination=0.05)

df['anomalia'] = iso.fit_predict(X_cluster)

plt.figure(figsize=(10,5))

plt.plot(df['cpu'], label='CPU')

anomalias = df[df['anomalia'] == -1]

plt.scatter(
    anomalias.index,
    anomalias['cpu']
)

plt.legend()

plt.title('Detecção de Anomalias')

plt.savefig('grafico_anomalias.png')

print('Anomalias detectadas:')
print(anomalias)