
import pyautogui
import pytesseract
from PIL import ImageGrab, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import subprocess
import logging

# === CONFIGURAR LOG DE DIAGNÓSTICO ===
logging.basicConfig(filename="ocr_diagnostico.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# === FUNÇÃO: DETECTA CAMINHO DO TESSERACT.EXE E VALIDA ===
def detectar_tesseract():
    caminho = r"C:\Users\Felipe Suckert\Desktop\3eye_leituradetela\3eye_tools\Tesseract-OCR\tesseract.exe"
    if os.path.exists(caminho):
        try:
            resultado = subprocess.run([caminho, "--version"], capture_output=True, text=True, timeout=5)
            if resultado.returncode == 0:
                logging.info(f"Tesseract encontrado: {resultado.stdout.strip()}")
                return caminho
            else:
                logging.error("Tesseract encontrado mas falhou ao executar.")
                raise RuntimeError("Tesseract está presente mas não responde corretamente.")
        except Exception as e:
            logging.exception("Erro ao validar execução do Tesseract:")
            raise RuntimeError(f"Erro ao validar Tesseract: {e}")
    else:
        logging.error("Tesseract.exe não encontrado no caminho especificado.")
        raise FileNotFoundError("❌ Tesseract.exe não encontrado. Verifique o caminho no script.")

# === INICIALIZAÇÃO ===
try:
    pytesseract.pytesseract.tesseract_cmd = detectar_tesseract()
    print(f"✅ Tesseract OK: {pytesseract.pytesseract.tesseract_cmd}")
except Exception as e:
    print(f"❌ ERRO: {e}")
    print("Veja detalhes em ocr_diagnostico.log")
    exit()

# CAPTURA A REGIÃO DO GRÁFICO — AJUSTE AS COORDENADAS SE PRECISAR
bbox = (100, 100, 1200, 700)

print("⏳ Aguardando 2 segundos para capturar a tela...")
time.sleep(2)

print("📸 Capturando gráfico com pyautogui...")
screenshot = pyautogui.screenshot(region=bbox)
screenshot_np = np.array(screenshot)

# CONVERTE A IMAGEM PARA ESCALA DE CINZA
gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

# EXTRAI TEXTO COM OCR
try:
    text = pytesseract.image_to_string(gray, lang='eng')
    logging.info("OCR executado com sucesso.")
except Exception as e:
    logging.exception("Erro ao executar OCR:")
    print(f"❌ Erro ao executar o OCR: {e}")
    exit()

# MOSTRA A IMAGEM NA TELA
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
plt.title("🧠 Gráfico capturado (OCR ativo)")
plt.axis("off")
plt.show()

# MOSTRA O TEXTO ENCONTRADO
print("\n📋 TEXTO DETECTADO NA TELA:")
print("=" * 50)
print(text)
print("=" * 50)
