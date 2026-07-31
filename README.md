# Telco360 CRM & MLOps Dashboard

Proyecto integral de Machine Learning con arquitectura orientada a producto SaaS. Diseñado para la segmentación de clientes de telecomunicaciones mediante aprendizaje no supervisado y la visualización de KPIs de negocio.

## Características Principales

*   **Portal CRM (Ventas):** Interfaz para agentes de retención. Permite ingresar datos de un cliente y obtener en tiempo real su segmento (VIP, Riesgo de Fuga, etc.) junto con recomendaciones accionables generadas por IA.
*   **Dashboard Directivo:** Panel de control de alto nivel (KPIs, ratios de LTV/CAC simulados, distribuciones) con un diseño *Craft-First* inspirado en interfaces premium (SaaS).
*   **Centro MLOps & Dev:** Consola de monitoreo en tiempo real de las métricas internas y externas del modelo (Silueta, Davies-Bouldin) y logs del sistema.
*   **Motor de IA (Backend):** API robusta desarrollada en **FastAPI** que ejecuta modelos de *K-Means* y *PCA* preentrenados y procesa la inferencia.

## Arquitectura del Proyecto

```text
Proyecto/
├── app/
│   ├── main.py                    # Backend FastAPI (API de Inferencia)
│   └── streamlit_app.py           # Frontend Streamlit (UI Premium)
├── data/                          # Datasets
├── metrics/                       # JSONs de evaluación de modelos
├── models/                        # Artefactos serializados (.pkl) 
│   ├── preprocessor.pkl
│   ├── pca_transformer.pkl
│   └── kmeans_model.pkl
├── scripts/
│   └── train_unsupervised.py      # Script de MLOps para entrenamiento
├── render.yaml                    # Infraestructura como Código (Blueprint)
└── requirements.txt               # Dependencias consolidadas
```

## Ejecución Local

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/rubenhuacasidev/telco360.git
    cd telco360
    ```
2.  **Entorno Virtual (Opcional pero recomendado):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Iniciar API (Backend):**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```
5.  **Iniciar Aplicación (Frontend):**
    ```bash
    streamlit run app/streamlit_app.py
    ```

## Despliegue en Render (En 1 Clic)

El proyecto incluye un archivo `render.yaml` preparado para Infraestructura como Código (IaC).
Para desplegar gratuitamente ambos servicios conectados automáticamente:

1. Ingresa al dashboard de [Render.com](https://render.com)
2. Haz clic en **New +** y selecciona **Blueprint**
3. Conecta tu cuenta de GitHub y elige este repositorio.
4. Render levantará 2 *Web Services* (API y UI) en paralelo, vinculando las variables de entorno automáticamente.

## Tecnologías Utilizadas
*   **Backend:** FastAPI, Uvicorn, Python 3.10+
*   **Frontend:** Streamlit, Altair (Visualización)
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **Diseño:** UI/UX optimizada con principios Craft-First.

---
*Proyecto académico — Aprendizaje Máquina, Semestre IX*
