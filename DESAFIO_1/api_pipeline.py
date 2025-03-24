from fastapi import FastAPI
from fastapi.responses import FileResponse

from ml_pipeline.etl import carregar_dados
from ml_pipeline.features import extrair_features_agregadas
from ml_pipeline.modelagem import treinar_modelos
from ml_pipeline.avaliacao import avaliar_modelo
from ml_pipeline.predicao import gerar_previsoes
from ml_pipeline.monitoramento import iniciar_mlflow_experimento

import pandas as pd
import mlflow
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(BASE_DIR, "dados")
PREVISAO_DIR = os.path.join(BASE_DIR, "previsao")
os.makedirs(PREVISAO_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(PREVISAO_DIR, "previsao_classificada.csv")

X_raw, y, X, resultados, scaler = None, None, None, None, None

@app.get("/etl")
def etl():
    global X_raw, y, X
    X_raw, y = carregar_dados()
    X = extrair_features_agregadas(X_raw)
    return {"message": "Dados carregados e features extraídas."}

@app.get("/treinar")
def treinar():
    global resultados, scaler, X, y
    if X is None or y is None:
        return {"error": "Rode /etl primeiro."}
    resultados, scaler = treinar_modelos(X.drop(columns=["series_id"]), y)
    return {"message": "Modelos treinados."}

@app.get("/avaliar")
def avaliar():
    global resultados
    if resultados is None:
        return {"error": "Rode /treinar primeiro."}
    for nome, res in resultados.items():
        print(f"\n=== {nome} ===")
        avaliar_modelo(res['y_test'], res['y_pred'])
        mlflow.log_metric(f"accuracy_{nome}", res['acuracia'])
    return {"message": "Avaliação finalizada."}

@app.get("/prever")
def prever():
    global resultados, scaler, X
    if resultados is None or scaler is None:
        return {"error": "Rode /treinar primeiro."}
    path_x_test = os.path.join(DADOS_DIR, "X_test.csv")
    X_test_raw = pd.read_csv(path_x_test)
    X_test_features = extrair_features_agregadas(X_test_raw)

    melhor_modelo = resultados['Random Forest']['modelo']
    feature_names = X.drop(columns=["series_id"]).columns.tolist()

    df_submission = gerar_previsoes(melhor_modelo, X_test_features, scaler, feature_names)
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    mlflow.log_artifact(SUBMISSION_PATH)
    return FileResponse(SUBMISSION_PATH, media_type='text/csv', filename='previsao_classificada.csv')

@app.get("/pipeline")
def run_pipeline():
    if mlflow.active_run():
        mlflow.end_run()
    iniciar_mlflow_experimento("careercon2019")
    etl()
    treinar()
    avaliar()
    prever()
    mlflow.end_run()
    return FileResponse(SUBMISSION_PATH, media_type='text/csv', filename='previsao_classificada.csv')