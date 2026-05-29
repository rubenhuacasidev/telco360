"""
test_api.py — Tests básicos para la API de predicción de Churn.

Ejecutar con: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------
VALID_CLIENT = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHealthEndpoints:
    """Tests para endpoints de health check."""

    def test_root(self):
        """GET / retorna status ok."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health(self):
        """GET /health retorna información del modelo."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["scaler_loaded"] is True
        assert data["features_count"] > 0


class TestPredictEndpoint:
    """Tests para el endpoint de predicción."""

    def test_predict_valid(self):
        """POST /predict con datos válidos retorna predicción."""
        response = client.post("/predict", json=VALID_CLIENT)
        assert response.status_code == 200
        data = response.json()
        assert "churn" in data
        assert "probabilidad" in data
        assert isinstance(data["churn"], bool)
        assert 0.0 <= data["probabilidad"] <= 1.0

    def test_predict_response_schema(self):
        """La respuesta tiene exactamente los campos esperados."""
        response = client.post("/predict", json=VALID_CLIENT)
        data = response.json()
        assert set(data.keys()) == {"churn", "probabilidad"}

    def test_predict_high_risk_client(self):
        """Cliente de alto riesgo vs. estable: la probabilidad debe ser distinta."""
        high_risk = {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0,
        }
        # Predicción del cliente de alto riesgo
        resp_high = client.post("/predict", json=high_risk)
        assert resp_high.status_code == 200
        prob_high = resp_high.json()["probabilidad"]

        # Predicción del cliente estable (VALID_CLIENT)
        resp_stable = client.post("/predict", json=VALID_CLIENT)
        assert resp_stable.status_code == 200
        prob_stable = resp_stable.json()["probabilidad"]

        # El cliente de alto riesgo debe tener mayor probabilidad de churn
        assert prob_high > prob_stable

    def test_predict_missing_field(self):
        """POST /predict sin un campo requerido retorna 422."""
        incomplete = VALID_CLIENT.copy()
        del incomplete["gender"]
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_invalid_tenure(self):
        """POST /predict con tenure negativo retorna 422."""
        invalid = VALID_CLIENT.copy()
        invalid["tenure"] = -5
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422
