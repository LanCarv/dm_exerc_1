import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def analise_por_coluna(df):
    return pd.DataFrame([{
        'Coluna': col,
        'Registros': df[col].count(),
        'Nulos': df[col].isnull().sum(),
        'Perc Nulos': (df[col].isnull().sum() / df.shape[0]) * 100,
        'Registro únicos': df[col].nunique(),
        'Valor mais frequente': df[col].mode()[0] if not df[col].mode().empty else None,
        'Frequência do valor mais comum': df[col].value_counts().max() if not df[col].value_counts().empty else None,
        'Tipo dado': df[col].dtype
    } for col in df.columns])