"""
train_unsupervised.py — Script de reentrenamiento automatizado para segmentación (K-Means).

Emula el notebook assignment4-unsupervised-segmentation.ipynb.
Calcula métricas reales y exporta modelos y métricas a disco.
"""

import os
import json
import pickle
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df

def train_and_evaluate():
    print("Iniciando pipeline de entrenamiento no supervisado...")
    df = load_data()
    
    # Seleccionamos las mismas features que en la app
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    X = df[features].copy()
    
    # Identificar columnas categóricas y numéricas
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in features if c not in num_cols]
    
    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    print("Transformando datos...")
    X_processed = preprocessor.fit_transform(X)
    
    # K-Means con k=4
    print("Entrenando modelo K-Means (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_processed)
    
    print("Calculando métricas de evaluación...")
    silhouette = silhouette_score(X_processed, cluster_labels)
    davies = davies_bouldin_score(X_processed, cluster_labels)
    calinski = calinski_harabasz_score(X_processed, cluster_labels)
    
    metrics = {
        "model": "K-Means",
        "hyperparameters": {"n_clusters": 4, "random_state": 42},
        "silhouette_score": round(silhouette, 4),
        "davies_bouldin_index": round(davies, 4),
        "calinski_harabasz_index": round(calinski, 4),
        "samples": len(X),
        "features_after_transform": X_processed.shape[1],
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Guardar clientes leales para el recomendador KNN
    print("Generando dataset de clientes leales...")
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    loyal_mask = (df_with_clusters['Churn'] == 'No') & (df_with_clusters['tenure'] >= 12)
    df_loyal = df_with_clusters[loyal_mask].copy()
    
    # Como quitamos NaNs, len(df) == len(X_processed).
    # Obtener boolean mask en forma de array:
    mask_array = loyal_mask.values
    loyal_vectors = X_processed[mask_array]
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    
    print("Guardando artefactos en disco...")
    
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
        
    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
        
    with open(os.path.join(MODELS_DIR, "loyal_vectors.pkl"), "wb") as f:
        pickle.dump(loyal_vectors, f)
        
    df_loyal.to_csv(os.path.join(BASE_DIR, "data", "loyal_customers.csv"), index=False)
    
    with open(os.path.join(METRICS_DIR, "unsupervised_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    history_path = os.path.join(METRICS_DIR, "training_history.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except Exception:
                pass
    history.insert(0, metrics)
    history = history[:10] # Guardar los últimos 10
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    print("¡Reentrenamiento No Supervisado Completado!")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    train_and_evaluate()
