
import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="📈 Dashboard 3eye Metrics", layout="wide")
st.title("📊 Histórico e Métricas - 3eye Vision Supreme")

backup_dir = "backups"
if not os.path.exists(backup_dir):
    st.warning("Nenhuma análise encontrada. Execute o sistema principal primeiro.")
else:
    files = sorted([f for f in os.listdir(backup_dir) if f.endswith(".json")], reverse=True)
    data = []

    for file in files:
        with open(os.path.join(backup_dir, file), "r", encoding="utf-8") as f:
            content = json.load(f)
            data.append(content)

    if data:
        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"])
        df = df.sort_values(by="data", ascending=False)
        st.dataframe(df)

        st.line_chart(df.set_index("data")["score"].astype(int))

        st.download_button("⬇️ Exportar CSV", data=df.to_csv(index=False), file_name="historico_3eye.csv")

        st.success(f"{len(df)} análises carregadas.")
    else:
        st.info("Nenhum dado disponível ainda.")
