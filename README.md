# 🧠 Projeto 1 - DM Case técnico exercício 1 - CareerCon 2019 (Classificação de Superfície)

Este projeto realiza a predição do tipo de superfície onde um robô se movimenta, com base em sensores de movimento. Foi desenvolvido para o desafio do Kaggle: [CareerCon 2019](https://www.kaggle.com/competitions/career-con-2019/overview).

O pipeline:
- 📦 ETL a partir de um `.zip` local
- 🛠️ Engenharia de features com agregações
- 🤖 Treinamento de múltiplos modelos (Logistic Regression, SVM, KNN, Random Forest)
- 📊 Avaliação e logging com **MLflow**
- 🔮 Geração de submissão para o Kaggle
- 🌐 **API com FastAPI** com endpoints para cada etapa

---

## ⚙️ Requisitos

- Python 3.10
- Ambiente virtual

Instale os requisitos:
```bash
pip install -r requirements.txt
```

---

## 📁 Estrutura de Pastas

```
DESAFIO_1/
├── api_pipeline.py         # API FastAPI com endpoints
├── main.py                 # Execução direta como script (opcional)
├── requirements.txt
├── .env                    # Caminho do ZIP (ZIP_PATH_1, EXTRACT_PATH_1)
│
├── dados/                  # Contém o .zip e CSVs extraídos
├── previsao/               # Previsões geradas
├── ml_pipeline/            # Módulos separados por etapa
│   ├── etl.py
│   ├── features.py
│   ├── modelagem.py
│   ├── avaliacao.py
│   ├── predicao.py
│   └── monitoramento.py
└── notebooks/              # EDA e protótipos
```

---

## 🚀 Como rodar o projeto (FastAPI)

### 1. Clone o repositório e crie o ambiente:
```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo/DESAFIO_1
python -m venv desafio_tecnico
source desafio_tecnico/Scripts/activate  # ou .\desafio_tecnico\Scripts\activate no Windows
```

### 2. Instale os pacotes:
```bash
pip install -r requirements.txt
```

### 3. Configure o `.env`
```env
ZIP_PATH_1=C:\caminho\para\career-con-2019.zip
EXTRACT_PATH_1=C:\caminho\para\extração\dos\csvs
```

### 4. Suba a API
```bash
uvicorn api_pipeline:app --reload
```

### 5. Acesse no navegador:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 Endpoints disponíveis

| Método | Rota         | Função                          |
|--------|--------------|----------------------------------|
| GET    | `/etl`       | Executa o ETL e extrai features  |
| GET    | `/treinar`   | Treina os modelos                |
| GET    | `/avaliar`   | Avalia os modelos e loga no MLflow |
| GET    | `/prever`    | Gera submissão e baixa CSV       |
| GET    | `/pipeline`  | Executa tudo e baixa as previsões no próprio Browser   |

---

## 📊 MLflow
Os experimentos são monitorados e logados automaticamente.
Para visualizar:
```bash
mlflow ui
```
Acesse: [http://localhost:5000](http://localhost:5000)

---

## 🔒 .gitignore sugerido
Crie um arquivo `.gitignore` na raiz com:

```
__pycache__/
*.pyc
.env
seu_ambiente_venv/
previsao/
mlruns/
*.zip
```

---

## 🙌 Créditos
- Desafio original: [Kaggle CareerCon 2019](https://www.kaggle.com/competitions/career-con-2019)
- Desenvolvido por: Luan de Carvalho Freitas - 23/03/2025 - 4 horas