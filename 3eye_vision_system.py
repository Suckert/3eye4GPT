# 3eye Vision — Sistema Autônomo Supremo de Trading Visual + Inteligência Total

import pyautogui
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import time
from PIL import ImageGrab
import openai
import os
from textblob import TextBlob
from perfil_3eye_vision import classify_trade, sentiment_from_news
from executor_3eye_core import obter_dados_binance
from pattern_detector_3eye import detectar_padroes
import matplotlib.pyplot as plt
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

openai.api_key = os.getenv("OPENAI_API_KEY")

# === Banco de Dados Local para Treinamento e Backtest ===
conn = sqlite3.connect("3eye_data.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
    timestamp TEXT,
    par TEXT,
    acao TEXT,
    sentimento REAL,
    modo TEXT,
    close REAL,
    score INTEGER,
    padroes TEXT
)''')
conn.commit()

# === Auto Detector de Coordenadas ===
def detectar_regioes():
    st.info("Mova o mouse para o canto superior esquerdo da região do gráfico e pressione ENTER.")
    if st.button("Capturar canto superior esquerdo"):
        p1 = pyautogui.position()
        time.sleep(1)
        st.info("Agora, mova o mouse para o canto inferior direito e pressione ENTER.")
        if st.button("Capturar canto inferior direito"):
            p2 = pyautogui.position()
            bbox = (p1.x, p1.y, p2.x, p2.y)
            st.success(f"Região definida: {bbox}")
            return bbox
    return None

# === Painel de Controle Streamlit ===
def painel_controle():
    st.set_page_config(page_title="3eye Vision AI Panel", layout="wide")
    st.title("🧿 3eye Vision - Painel Supremo de Controle")
    col1, col2 = st.columns(2)

    with col1:
        modo = st.selectbox("Modo de Operação", ["Assistido", "Livre", "Alavancado", "Visual Prophet"])
        moedas = st.multiselect("Ativos Monitorados", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"], default=["BTC/USDT"])
        intervalo = st.slider("Intervalo (segundos entre análises)", 1, 60, 10)

    with col2:
        st.metric(label="📡 Status do Sistema", value="Online")
        st.info("Configure os parâmetros antes de iniciar.")

    if modo == "Visual Prophet":
        bbox = detectar_regioes()
        if bbox:
            img = np.array(ImageGrab.grab(bbox=bbox))
            st.image(img, caption="Região capturada")
    
    if st.button("🚀 Iniciar Sistema"):
        st.success(f"Sistema Iniciado em modo: {modo} com {moedas}")
        iniciar_monitoramento(moedas, modo, intervalo)

    st.markdown("---")
    st.subheader("📊 Histórico de Decisões")
    mostrar_historico()

    st.markdown("---")
    st.subheader("🧠 Auto-Treinamento Inteligente")
    if st.button("🔁 Treinar com Histórico"):
        treinar_modelo()

    st.markdown("---")
    st.subheader("🧪 Simulação com Replay de Sinais")
    if st.button("🎥 Iniciar Backtest com Replay"):
        executar_simulacao()

# === Execução ===
def iniciar_monitoramento(moedas, modo, intervalo):
    for par in moedas:
        df = obter_dados_binance(par)
        noticia = f"{par} movimentação incomum hoje"  # placeholder
        sentimento = sentiment_from_news(noticia) if modo != "Livre" else gpt_sentiment(noticia)
        acao = classify_trade(df, sentimento)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        score = calcular_score(acao)
        close = df['close'].iloc[-1]
        padroes = detectar_padroes(df)
        descricoes = ', '.join([p[1] for p in padroes]) if padroes else 'Nenhum'
        cursor.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)", (timestamp, par, acao, sentimento, modo, close, score, descricoes))
        conn.commit()
        st.write(f"{timestamp} | {par} | Decisão: {acao.upper()} | Sentimento: {sentimento:.2f} | Close: {close:.2f} | Score: {score} | Padrões: {descricoes}")
        time.sleep(intervalo)

# === GPT Análise Livre ===
def gpt_sentiment(texto):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um analista de sentimento de mercado financeiro."},
            {"role": "user", "content": texto}
        ]
    )
    return TextBlob(response['choices'][0]['message']['content']).sentiment.polarity

# === Score Simples ===
def calcular_score(acao):
    return {"buy": 3, "hold": 2, "sell": 1}.get(acao, 0)

# === Histórico Visual ===
def mostrar_historico():
    df_hist = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100", conn)
    st.dataframe(df_hist)
    if not df_hist.empty:
        st.line_chart(df_hist[['timestamp', 'score']].set_index('timestamp'))

# === Auto Treinamento ===
def treinar_modelo():
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    if len(df) < 10:
        st.warning("Poucos dados para treinar. Execute mais sinais antes.")
        return
    df['acao'] = df['acao'].map({'buy': 1, 'hold': 0, 'sell': -1})
    df.dropna(inplace=True)
    X = df[['sentimento', 'score']]
    y = df['acao']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.success(f"Modelo treinado com acurácia: {acc:.2f}")
    st.code(classification_report(y_test, model.predict(X_test)))

# === Simulação de Replay ===
def executar_simulacao():
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp ASC", conn)
    if df.empty:
        st.warning("Nenhum dado para simular. Execute operações antes.")
        return
    speed = st.slider("Velocidade de Replay (seg)", 0.1, 5.0, 1.0)
    st.write("🎬 Iniciando Replay de Sinais")
    for _, row in df.iterrows():
        st.info(f"{row['timestamp']} | {row['par']} | {row['acao'].upper()} | Sent: {row['sentimento']:.2f} | Close: {row['close']} | Score: {row['score']} | {row['padroes']}")
        time.sleep(speed)

# === Rodar o Painel ===
painel_controle()
