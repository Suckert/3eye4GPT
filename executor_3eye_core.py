# executor_3eye_core.py
# Módulo central do 3eye Executor com IA livre, alavancagem automática, e 6 canais simultâneos

import os
import ccxt
import openai
import asyncio
import time
import datetime as dt
from dotenv import load_dotenv
from perfil_3eye_vision import classify_trade, sentiment_from_news

load_dotenv()

# === Configuração de Ambiente ===
openai.api_key = os.getenv("OPENAI_API_KEY")
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

binance = ccxt.binance({
    'apiKey': BINANCE_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}  # Ativa futuros para alavancagem
})

# === Configurações ===
MOEDAS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
ALAVANCAGEM = 5  # padrão, regulável por IA
INTERVALO = 1  # segundos entre execuções por slot
MODOS = ['assistido', 'livre', 'alavancado']

# === Funções auxiliares ===
def obter_dados_binance(par):
    ohlcv = binance.fetch_ohlcv(par, timeframe='1m', limit=100)
    df = binance.parse_ohlcv(ohlcv, market=par)
    df = [dict(zip(['timestamp', 'open', 'high', 'low', 'close', 'vol'], x)) for x in ohlcv]
    return pd.DataFrame(df)

def obter_sentimento_via_gpt(noticia):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um analista de sentimento de mercado."},
            {"role": "user", "content": noticia}
        ]
    )
    return float(TextBlob(response['choices'][0]['message']['content']).sentiment.polarity)

async def executar_slot(par, modo='assistido'):
    while True:
        try:
            df = obter_dados_binance(par)
            noticia = f"{par} sobe após volume incomum"  # Placeholder
            sentimento = obter_sentimento_via_gpt(noticia) if modo == 'livre' else sentiment_from_news(noticia)
            acao = classify_trade(df, sentimento)

            print(f"{dt.datetime.now()} | {par} | Ação: {acao.upper()} | Sentimento: {sentimento:.2f} | Modo: {modo}")

            if acao == 'buy':
                tamanho = 10  # USDT fixo por operação (ajustável)
                if modo == 'alavancado':
                    binance.set_leverage(ALAVANCAGEM, par)
                binance.create_market_buy_order(par, tamanho / df['close'].iloc[-1])

            elif acao == 'sell':
                tamanho = 10
                if modo == 'alavancado':
                    binance.set_leverage(ALAVANCAGEM, par)
                binance.create_market_sell_order(par, tamanho / df['close'].iloc[-1])

        except Exception as e:
            print(f"Erro no par {par}: {e}")

        await asyncio.sleep(INTERVALO)

# === Execução Principal ===
async def main():
    tarefas = [executar_slot(par, modo='assistido') for par in MOEDAS]
    await asyncio.gather(*tarefas)

if __name__ == "__main__":
    asyncio.run(main())
