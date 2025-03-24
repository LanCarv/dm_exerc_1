import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def extrair_features_agregadas(df, group_col='series_id'):
    sensor_cols = [col for col in df.columns if col not in ['row_id', 'series_id', 'measurement_number']]
    agg_funcs = ['mean', 'std', 'min', 'max', 'median', 'skew']
    df_features = df.groupby(group_col)[sensor_cols].agg(agg_funcs)
    df_features.columns = ['_'.join(col) for col in df_features.columns]
    df_features.reset_index(inplace=True)
    df_features = df_features.fillna(0)
    return df_features