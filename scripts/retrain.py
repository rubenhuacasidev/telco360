"""
retrain.py — Script de reentrenamiento automatizado.

Reproduce la lógica del notebook para entrenar un modelo de Logistic Regression
con los mejores hiperparámetros encontrados (C=10, solver='lbfgs').

Genera:
  - models/model.pkl   → Modelo entrenado
  - models/scaler.pkl  → Scaler entrenado
  - metrics/metrics.json → Métricas de evaluación
"""

import os
import json
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Configuración de rutas (relativas a la raíz del proyecto)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")


def load_and_clean(path: str) -> pd.DataFrame:
    """Carga el dataset y aplica limpieza básica."""
    df = pd.read_csv(path)

    # Convertir TotalCharges a numérico (hay espacios vacíos)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Eliminar customerID (no es útil para predicción)
    df.drop("customerID", axis=1, inplace=True)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica la variable objetivo y las categóricas."""
    # Codificar Churn a binario
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encoding de variables categóricas
    df = pd.get_dummies(df, drop_first=True)

    return df


def train_model(df: pd.DataFrame):
    """Entrena Logistic Regression y retorna modelo, scaler, métricas."""
    # Separar features y target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # División 80/20 estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento con los mejores hiperparámetros del notebook
    # (C=10, solver='lbfgs') → mejor F1-Score en GridSearchCV
    model = LogisticRegression(
        C=10, solver="lbfgs", max_iter=1000, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Métricas
    metrics = {
        "model": "LogisticRegression",
        "hyperparameters": {"C": 10, "solver": "lbfgs", "max_iter": 1000},
        "accuracy": round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred), 6),
        "recall": round(recall_score(y_test, y_pred), 6),
        "f1_score": round(f1_score(y_test, y_pred), 6),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 6),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": int(X_train.shape[1]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Guardar columnas esperadas para validación en la API
    feature_columns = list(X.columns)

    return model, scaler, metrics, feature_columns


def save_artifacts(model, scaler, metrics: dict, feature_columns: list):
    """Guarda modelo, scaler y métricas en disco."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    columns_path = os.path.join(MODELS_DIR, "feature_columns.json")
    metrics_path = os.path.join(METRICS_DIR, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Modelo guardado en: {model_path}")
    print(f"  ✓ Scaler guardado en: {scaler_path}")
    print(f"  ✓ Columnas guardadas en: {columns_path}")
    print(f"  ✓ Métricas guardadas en: {metrics_path}")


def main():
    print("=" * 60)
    print("PIPELINE DE REENTRENAMIENTO — Telco Customer Churn")
    print("=" * 60)

    # 1. Cargar y limpiar
    print("\n[1/4] Cargando y limpiando datos...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: No se encontró el dataset en {DATA_PATH}")
        sys.exit(1)
    df = load_and_clean(DATA_PATH)
    print(f"  → {df.shape[0]} filas, {df.shape[1]} columnas después de limpieza")

    # 2. Codificar
    print("\n[2/4] Codificando variables...")
    df = encode(df)
    print(f"  → {df.shape[1]} features después de one-hot encoding")

    # 3. Entrenar
    print("\n[3/4] Entrenando modelo (LogisticRegression, C=10, lbfgs)...")
    model, scaler, metrics, feature_columns = train_model(df)
    print(f"  → Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  → Precision: {metrics['precision']:.4f}")
    print(f"  → Recall:    {metrics['recall']:.4f}")
    print(f"  → F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  → ROC-AUC:   {metrics['roc_auc']:.4f}")

    # 4. Guardar
    print("\n[4/4] Guardando artefactos...")
    save_artifacts(model, scaler, metrics, feature_columns)

    print("\n" + "=" * 60)
    print("REENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()
