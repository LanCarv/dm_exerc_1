import mlflow
import warnings
warnings.filterwarnings("ignore")

def iniciar_mlflow_experimento(nome_experimento):
    mlflow.set_experiment(nome_experimento)
    mlflow.start_run()