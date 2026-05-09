import psutil
import pandas as pd
import time
from datetime import datetime
import os

arquivo = 'dados_monitorados.csv'

while True:

    dados = {
        'timestamp': [datetime.now()],
        'cpu': [psutil.cpu_percent()],
        'memoria': [psutil.virtual_memory().percent],
        'disco': [psutil.disk_usage('/').percent]
    }

    df_novo = pd.DataFrame(dados)

    if os.path.exists(arquivo):
        df_antigo = pd.read_csv(arquivo)
        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_csv(arquivo, index=False)

    print(df_novo)

    time.sleep(5)