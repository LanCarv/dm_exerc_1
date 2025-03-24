import warnings
warnings.filterwarnings("ignore")

def gerar_previsoes(modelo, df_test_features, scaler, feature_names):
    X_input = df_test_features.drop(columns=["series_id"], errors='ignore')
    X_input = X_input.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X_input)
    y_pred = modelo.predict(X_scaled)
    df_test_features['surface'] = y_pred
    return df_test_features[['series_id', 'surface']]