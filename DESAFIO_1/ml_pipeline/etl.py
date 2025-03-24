import pandas as pd
import zipfile
import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

def carregar_dados():
    # Carrega .env
    load_dotenv()

    # Caminhos do .env
    zip_path = os.getenv("ZIP_PATH_1")
    extract_path = os.getenv("EXTRACT_PATH_1")

    # Extrai se ainda não estiver extraído
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    if not any(fname.endswith(".csv") for fname in os.listdir(extract_path)):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # Carrega os arquivos
    path_x = os.path.join(extract_path, "X_train.csv")
    path_y = os.path.join(extract_path, "y_train.csv")

    X = pd.read_csv(path_x)
    y_df = pd.read_csv(path_y)

    # ✅ AQUI ESTÁ O PONTO CRÍTICO
    y = y_df["surface"]  # Garante que é uma Series apenas com o alvo

    return X, y