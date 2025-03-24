import os
import pandas as pd
import mlflow

from ml_pipeline.etl import carregar_dados
from ml_pipeline.features import extrair_features_agregadas
from ml_pipeline.modelagem import treinar_modelos
from ml_pipeline.avaliacao import avaliar_modelo
from ml_pipeline.predicao import gerar_previsoes
from ml_pipeline.monitoramento import iniciar_mlflow_experimento

import warnings
warnings.filterwarnings("ignore")

# === Caminhos ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(BASE_DIR, "dados")
PREVISAO_DIR = os.path.join(BASE_DIR, "previsao")
os.makedirs(PREVISAO_DIR, exist_ok=True)

PATH_X_TRAIN = os.path.join(DADOS_DIR, "X_train.csv")
PATH_Y_TRAIN = os.path.join(DADOS_DIR, "y_train.csv")
PATH_X_TEST = os.path.join(DADOS_DIR, "X_test.csv")
SUBMISSION_PATH = os.path.join(PREVISAO_DIR, "submission.csv")

# === Pipeline ===
print("\n Iniciando experimento com MLflow")

# Garante que não há run ativa
if mlflow.active_run():
    mlflow.end_run()

# Inicia experimento e run
iniciar_mlflow_experimento("careercon2019")

# 1. Leitura dos dados
print("\n Carregando dados...")
X_raw, y = carregar_dados()

# 2. Engenharia de features
print("\n🛠️  Extraindo features agregadas...")
X = extrair_features_agregadas(X_raw)

# 3. Modelagem
print("\n Treinando modelos...")
resultados, scaler = treinar_modelos(X.drop(columns=["series_id"]), y)

# 4. Avaliando resultados
print("\n Avaliando modelos...")
for nome, res in resultados.items():
    print(f"\n=== {nome} ===")
    avaliar_modelo(res['y_test'], res['y_pred'])
    mlflow.log_metric(f"accuracy_{nome}", res['acuracia'])

# 5. Escolhe melhor modelo (Random Forest aqui)
melhor_modelo = resultados['Random Forest']['modelo']
feature_names = X.drop(columns=["series_id"]).columns.tolist()

# 6. Gera previsões para o X_test
print("\n Gerando submissão para X_test.csv...")
X_test_raw = pd.read_csv(PATH_X_TEST)
X_test_features = extrair_features_agregadas(X_test_raw)
df_submission = gerar_previsoes(melhor_modelo, X_test_features, scaler, feature_names)

# 7. Salva CSV
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f" Submissão salva em: {SUBMISSION_PATH}")

# 8. Log de artefato
mlflow.log_artifact(SUBMISSION_PATH)
print("\n Experimento finalizado com sucesso!")

# 9. Encerra run
mlflow.end_run()
