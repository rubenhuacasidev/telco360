"""
main.py — API de predicción de Churn con FastAPI.

Endpoints:
  GET  /         → Health check
  GET  /health   → Estado del modelo
  POST /predict  → Predicción de churn para un cliente
"""

import os
import json

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(BASE_DIR, "metrics", "metrics.json")

# ---------------------------------------------------------------------------
# Cargar artefactos al inicio
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

model = None
scaler = None
feature_columns = None
metrics_data = None


def load_artifacts():
    """Carga modelo, scaler, columnas y métricas."""
    global model, scaler, feature_columns, metrics_data

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler no encontrado: {SCALER_PATH}")
    if not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError(f"Columnas no encontradas: {COLUMNS_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)


# Cargar al importar el módulo
load_artifacts()


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predecir la fuga de clientes de telecomunicaciones",
    version="1.0.0",
)

# CORS — permitir que Streamlit se conecte desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------
class ClienteInput(BaseModel):
    """Datos de entrada del cliente (mismos campos del dataset)."""

    gender: str = Field(..., description="Género: 'Male' o 'Female'")
    SeniorCitizen: int = Field(..., description="Ciudadano senior: 0 o 1")
    Partner: str = Field(..., description="Tiene pareja: 'Yes' o 'No'")
    Dependents: str = Field(..., description="Tiene dependientes: 'Yes' o 'No'")
    tenure: int = Field(..., description="Meses como cliente", ge=0)
    PhoneService: str = Field(..., description="Servicio telefónico: 'Yes' o 'No'")
    MultipleLines: str = Field(
        ..., description="Líneas múltiples: 'Yes', 'No' o 'No phone service'"
    )
    InternetService: str = Field(
        ..., description="Servicio de internet: 'DSL', 'Fiber optic' o 'No'"
    )
    OnlineSecurity: str = Field(
        ..., description="Seguridad en línea: 'Yes', 'No' o 'No internet service'"
    )
    OnlineBackup: str = Field(
        ..., description="Backup en línea: 'Yes', 'No' o 'No internet service'"
    )
    DeviceProtection: str = Field(
        ...,
        description="Protección de dispositivo: 'Yes', 'No' o 'No internet service'",
    )
    TechSupport: str = Field(
        ..., description="Soporte técnico: 'Yes', 'No' o 'No internet service'"
    )
    StreamingTV: str = Field(
        ..., description="Streaming TV: 'Yes', 'No' o 'No internet service'"
    )
    StreamingMovies: str = Field(
        ..., description="Streaming Movies: 'Yes', 'No' o 'No internet service'"
    )
    Contract: str = Field(
        ...,
        description="Tipo de contrato: 'Month-to-month', 'One year' o 'Two year'",
    )
    PaperlessBilling: str = Field(
        ..., description="Facturación sin papel: 'Yes' o 'No'"
    )
    PaymentMethod: str = Field(
        ...,
        description="Método de pago: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)' o 'Credit card (automatic)'",
    )
    MonthlyCharges: float = Field(..., description="Cargos mensuales", ge=0)
    TotalCharges: float = Field(..., description="Cargos totales", ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    }


class PrediccionOutput(BaseModel):
    """Resultado de la predicción."""

    churn: bool = Field(..., description="¿El cliente se va?")
    probabilidad: float = Field(
        ..., description="Probabilidad de churn (0.0 a 1.0)"
    )


# ---------------------------------------------------------------------------
# Función de preprocesamiento
# ---------------------------------------------------------------------------
def preprocess_input(data: ClienteInput) -> np.ndarray:
    """
    Convierte los datos del cliente al formato esperado por el modelo.

    No usa pd.get_dummies (falla con una sola fila + drop_first).
    En su lugar, construye manualmente el vector de features alineado
    con las columnas del entrenamiento.
    """
    input_dict = data.model_dump()

    # Inicializar todas las columnas en 0
    row = {col: 0 for col in feature_columns}

    # --- Columnas numéricas directas ---
    row["SeniorCitizen"] = input_dict["SeniorCitizen"]
    row["tenure"] = input_dict["tenure"]
    row["MonthlyCharges"] = input_dict["MonthlyCharges"]
    row["TotalCharges"] = input_dict["TotalCharges"]

    # --- Columnas dummy (drop_first: la primera categoría alfabética se omite) ---
    # Mapa: campo → { valor_original → columna_dummy }
    # Solo las categorías que NO son la primera (drop_first) tienen columna
    dummy_map = {
        "gender": {"Male": "gender_Male"},
        "Partner": {"Yes": "Partner_Yes"},
        "Dependents": {"Yes": "Dependents_Yes"},
        "PhoneService": {"Yes": "PhoneService_Yes"},
        "MultipleLines": {
            "No phone service": "MultipleLines_No phone service",
            "Yes": "MultipleLines_Yes",
        },
        "InternetService": {
            "Fiber optic": "InternetService_Fiber optic",
            "No": "InternetService_No",
        },
        "OnlineSecurity": {
            "No internet service": "OnlineSecurity_No internet service",
            "Yes": "OnlineSecurity_Yes",
        },
        "OnlineBackup": {
            "No internet service": "OnlineBackup_No internet service",
            "Yes": "OnlineBackup_Yes",
        },
        "DeviceProtection": {
            "No internet service": "DeviceProtection_No internet service",
            "Yes": "DeviceProtection_Yes",
        },
        "TechSupport": {
            "No internet service": "TechSupport_No internet service",
            "Yes": "TechSupport_Yes",
        },
        "StreamingTV": {
            "No internet service": "StreamingTV_No internet service",
            "Yes": "StreamingTV_Yes",
        },
        "StreamingMovies": {
            "No internet service": "StreamingMovies_No internet service",
            "Yes": "StreamingMovies_Yes",
        },
        "Contract": {
            "One year": "Contract_One year",
            "Two year": "Contract_Two year",
        },
        "PaperlessBilling": {"Yes": "PaperlessBilling_Yes"},
        "PaymentMethod": {
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check",
        },
    }

    for field, mapping in dummy_map.items():
        value = input_dict[field]
        if value in mapping:
            col_name = mapping[value]
            if col_name in row:
                row[col_name] = 1

    # Construir DataFrame con el orden correcto
    df = pd.DataFrame([row], columns=feature_columns)

    # Escalar
    scaled = scaler.transform(df)

    return scaled


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    """Health check básico."""
    return {
        "status": "ok",
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    """Estado detallado del modelo cargado."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    response = {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_count": len(feature_columns) if feature_columns else 0,
    }

    if metrics_data:
        response["metrics"] = {
            "accuracy": metrics_data.get("accuracy"),
            "f1_score": metrics_data.get("f1_score"),
            "roc_auc": metrics_data.get("roc_auc"),
            "trained_at": metrics_data.get("trained_at"),
        }

    return response


@app.post("/predict", response_model=PrediccionOutput, tags=["Predicción"])
def predict(cliente: ClienteInput):
    """
    Predice si un cliente hará churn.

    Recibe los datos del cliente y retorna:
    - **churn**: booleano indicando si el cliente se va
    - **probabilidad**: probabilidad de churn (0.0 a 1.0)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        # Preprocesar
        X = preprocess_input(cliente)

        # Predecir
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        return PrediccionOutput(
            churn=bool(prediction),
            probabilidad=round(float(probability), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")
