import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Monitoramento Inteligente')

df = pd.read_csv('dados_monitorados.csv')

# -----------------------------
# CPU
# -----------------------------

st.subheader('Uso de CPU')

fig1, ax1 = plt.subplots()

ax1.plot(df['cpu'])

ax1.set_ylabel('CPU %')

st.pyplot(fig1)

# -----------------------------
# MEMÓRIA
# -----------------------------

st.subheader('Uso de Memória')

fig2, ax2 = plt.subplots()

ax2.plot(df['memoria'])

ax2.set_ylabel('Memória %')

st.pyplot(fig2)

# -----------------------------
# ANOMALIAS
# -----------------------------

if 'anomalia' in df.columns:

    st.subheader('Anomalias Detectadas')

    anomalias = df[df['anomalia'] == -1]

    st.write(anomalias)