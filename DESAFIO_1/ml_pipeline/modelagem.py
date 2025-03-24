import warnings
warnings.filterwarnings("ignore")

def treinar_modelos(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Remove classes com apenas 1 ocorrência
    classe_counts = y.value_counts()
    classes_validas = classe_counts[classe_counts > 1].index
    mask = y.isin(classes_validas)

    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print("➡️ Verificando NaNs antes do split...")
    print("NaNs em X:", X.isnull().sum().sum())
    print("NaNs em y:", y.isnull().sum())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelos
    modelos = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM Linear': SVC(kernel='linear'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train_scaled if nome != 'Random Forest' else X_train, y_train)
        y_pred = modelo.predict(X_test_scaled if nome != 'Random Forest' else X_test)
        resultados[nome] = {
            'modelo': modelo,
            'acuracia': (y_test == y_pred).mean(),
            'y_pred': y_pred,
            'y_test': y_test
        }

    return resultados, scaler
