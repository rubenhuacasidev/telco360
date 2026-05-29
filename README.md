# Telco Customer Churn — MLOps Pipeline

Proyecto de Machine Learning con arquitectura MLOps para predecir la fuga de clientes (**Churn**) en una empresa de telecomunicaciones.

## Arquitectura

```
notebook (experimentación)
    → retrain.py (entrenamiento automático)
    → model.pkl + scaler.pkl (artefactos)
    → FastAPI (API de predicciones)
    → Streamlit (interfaz visual)
    → Docker (2 contenedores)
    → Railway/Render (despliegue en la nube)
    → GitHub Actions (CI/CD + autoentrenamiento)
```

## Estructura del Proyecto

```
Proyecto/
├── data/                          # Dataset original
├── models/                        # Artefactos de modelo (.pkl)
├── notebooks/                     # Notebook de experimentación
├── app/
│   ├── main.py                    # API FastAPI
│   └── streamlit_app.py           # Frontend Streamlit
├── scripts/
│   └── retrain.py                 # Script de reentrenamiento
├── tests/
│   └── test_api.py                # Tests de la API
├── .github/workflows/
│   └── ci_cd.yml                  # Pipeline CI/CD
├── metrics/
│   └── metrics.json               # Métricas del modelo
├── Dockerfile                     # Docker para la API
├── Dockerfile.streamlit           # Docker para el Frontend
├── requirements.txt               # Dependencias de producción
├── requirements-dev.txt           # Dependencias de desarrollo
└── README.md
```

## Instalación Local

### Prerrequisitos
- Python 3.11+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd Proyecto

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements-dev.txt

# 4. Entrenar el modelo (genera models/ y metrics/)
python scripts/retrain.py

# 5. Ejecutar la API
uvicorn app.main:app --reload

# 6. En otra terminal, ejecutar Streamlit
streamlit run app/streamlit_app.py
```

### Acceso local
- **API FastAPI**: http://localhost:8000
- **Documentación API (Swagger)**: http://localhost:8000/docs
- **Streamlit Frontend**: http://localhost:8501

## Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Estado del modelo y métricas |
| `POST` | `/predict` | Predicción de churn |

### Ejemplo de uso (curl)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 29.85
  }'
```

Respuesta:
```json
{
  "churn": false,
  "probabilidad": 0.4123
}
```

## Docker

### API (FastAPI)
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Frontend (Streamlit)
```bash
docker build -f Dockerfile.streamlit -t churn-frontend .
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 churn-frontend
```

## Despliegue en Railway/Render

Se crean **2 servicios** desde el mismo repositorio:

1. **Servicio API** → Dockerfile: `Dockerfile`
2. **Servicio Frontend** → Dockerfile: `Dockerfile.streamlit`

En el servicio Frontend, configurar la variable de entorno:
```
API_URL=https://<tu-api>.up.railway.app
```

## CI/CD con GitHub Actions

El pipeline se ejecuta automáticamente:

1. **En cada push**: ejecuta tests
2. **En push a `main`**: reentrena el modelo, commitea artefactos, y Railway/Render hace redeploy automático

## Tests

```bash
pytest tests/ -v
```

## Modelo

- **Algoritmo**: Logistic Regression
- **Hiperparámetros**: C=10, solver='lbfgs' (optimizados con GridSearchCV en el notebook)
- **Métricas**: ~80% accuracy, ~0.60 F1-Score, ~0.83 ROC-AUC

## Autor

Proyecto académico — Aprendizaje Máquina, Semestre IX
