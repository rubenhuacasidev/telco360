# Informe Técnico y de Arquitectura: Telco360 CRM & MLOps Dashboard

**Fecha de Elaboración:** Julio 2026
**Propósito del Documento:** Proveer una guía técnica, de arquitectura e integración de la plataforma Telco360 orientada a los equipos de TI, MLOps y Desarrollo.

---

## 1. Visión General de la Arquitectura

Telco360 es una aplicación de Machine Learning (ML) enfocada a producto SaaS. Permite realizar segmentación no supervisada de clientes de telecomunicaciones y ofrecer recomendaciones de retención utilizando motores de inferencia. La plataforma adopta un enfoque de microservicios desacoplados: un **Backend (API de Inferencia)** y un **Frontend (UI de Interacción y Dashboards)**, orquestados mediante un pipeline de MLOps automatizado.

### Topología del Sistema

```text
[Cliente / Agente] ---> [Frontend (Streamlit)] --- (HTTP/REST) ---> [Backend (FastAPI)] ---> [Modelos Serializados]
                               |                                                                ^
                               v                                                                |
                        [Altair (Dashboards)]                                                   |
                                                                                        (Pipeline MLOps)
                                                                                        [train_unsupervised.py]
```

## 2. Componentes Técnicos y Repositorio

El proyecto tiene una estructura monorepo que alberga tanto el frontend, el backend, los artefactos del modelo y los scripts de entrenamiento continuo:

- `app/main.py`: Backend basado en **FastAPI**. Aloja los endpoints de inferencia de Machine Learning y Healthchecks.
- `app/streamlit_app.py`: Frontend construido en **Streamlit**. Presenta una UI *Craft-First*, dashboards interactivos, y actúa como cliente HTTP del backend.
- `scripts/train_unsupervised.py`: Pipeline de entrenamiento automatizado que transforma datos en crudo en los modelos listos para producción.
- `models/`: Directorio que contiene los pesos serializados (archivos `.pkl`).
- `metrics/`: Almacenamiento en formato JSON (`unsupervised_metrics.json`, `training_history.json`) para tracking de experimentos y calidad del modelo en producción.
- `render.yaml`: Blueprint de Infraestructura como Código (IaC) para orquestación en la nube (Render).

---

## 3. Especificaciones del Backend (Motor de IA)

El backend es un servicio asíncrono y de alto rendimiento que levanta los modelos en memoria al inicializarse. 

### Tecnologías:
- Python 3.10+
- FastAPI (Web framework)
- Uvicorn (Servidor ASGI)
- Scikit-Learn, Pandas, NumPy (Motor de inferencia y procesamiento matricial)

### Endpoints Disponibles:

#### 1. `POST /segment`
- **Función:** Asigna un cliente a un clúster utilizando el modelo **K-Means (k=4)**.
- **Entrada:** JSON con 19 features del cliente (Ej. `tenure`, `MonthlyCharges`, `Contract`, `InternetService`, etc.).
- **Proceso:** 
  1. Parsea el payload mediante `Pydantic`.
  2. Pasa los datos por el transformador serializado (`preprocessor.pkl`).
  3. Ejecuta `kmeans.predict()`.
- **Salida:** ID del clúster (0-3) y el nombre/descripción comercial del perfil (Ahorradores, VIP, Leales, En Riesgo de Fuga).

#### 2. `POST /recommend`
- **Función:** Sistema de recomendación basado en Filtrado Colaborativo usando **K-Nearest Neighbors (KNN)**.
- **Entrada:** Mismo JSON que `/segment`.
- **Proceso:**
  1. Identifica a qué clúster pertenece el cliente.
  2. Filtra una matriz precargada de "Clientes Leales" (`loyal_vectors.pkl`) para ese mismo clúster.
  3. Calcula distancias euclidianas para encontrar los $k=5$ vecinos más similares.
  4. Analiza los servicios adquiridos por estos vecinos que el cliente actual *no* tiene.
  5. Aplica reglas de negocio duras (Ej. Si el contrato es mensual, la recomendación principal forzada es un "Upgrade a Contrato de 1 Año" para retención).
- **Salida:** Lista de servicios recomendados, justificación estadística e impacto mensual estimado (Upsell / MRR extra).

#### 3. `GET /mlops/health`
- **Función:** Telemetría y monitoreo de la integridad de los artefactos del modelo.
- **Salida:** Estado de salud del modelo (Healthy / Degraded), timestamp de la última modificación del `.pkl`, tamaño en KB, versión y algoritmo.

---

## 4. Especificaciones del Frontend (UI & Dashboards)

La aplicación de interfaz, escrita en **Streamlit**, está dividida en 3 módulos lógicos accesibles mediante navegación lateral.

### Módulos UI:
1. **Portal CRM (Ventas):** Interfaz para Customer Success. Lee los registros locales y permite "simular" clientes mediante un formulario reactivo. Se comunica vía API REST a `API_URL`. Renderiza componentes visuales avanzados, gamificación de clústeres y gráficas de proyección financiera (LTV) usando Altair.
2. **Dashboard Directivo:** Panel de Business Intelligence que carga los datos locales, les aplica el algoritmo de segmentación cargado en memoria, y expone visualizaciones analíticas de negocio (100% Stacked Bar Charts, Gráficos de Dispersión para Customer Journey).
3. **Centro MLOps & Dev:** Consola para ingenieros de ML y DevOps.
   - Consulta el endpoint `/mlops/health`.
   - Lee métricas guardadas en disco en `metrics/`.
   - Visualiza comparativas de algoritmos de clustering usando PCA en 2D.
   - Emula un *Trigger* de CI/CD para reentrenamiento simulado.

---

## 5. Pipeline de MLOps y Ciclo de Vida del Modelo

El archivo `scripts/train_unsupervised.py` gestiona el reentrenamiento offline (batch) del ecosistema.

### Pasos del Pipeline:
1. **Ingesta de Datos:** Carga de `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
2. **Transformación (ETL):** `ColumnTransformer` que aplica `StandardScaler` a características numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) y `OneHotEncoder` a categóricas.
3. **Entrenamiento de K-Means:** Inicializa K-Means con $k=4$, fijando una semilla aleatoria para reproductibilidad.
4. **Evaluación Continua:** Calcula automáticamente:
   - *Silhouette Score* (Mide densidad intra-cluster vs distancia inter-cluster).
   - *Davies-Bouldin Index* (Mide la separación entre los clusters).
   - *Calinski-Harabasz Index* (Ratio de varianza).
5. **Generación de Corpus KNN:** Genera y exporta vectores matriciales (`loyal_vectors.pkl`) y CSV (`loyal_customers.csv`) solo para los clientes que no han hecho Churn y tienen > 12 meses de antigüedad.
6. **Publicación de Artefactos:** 
   - Sobrescribe `.pkl` en el directorio `models/`.
   - Actualiza el historial del modelo en `metrics/training_history.json`.

---

## 6. Integración y Despliegue (CI/CD / IaC)

El proyecto soporta **Infraestructura como Código (IaC)** mediante `render.yaml`. 

Al integrarse con Render.com (o adaptándolo a entornos tipo AWS ECS / Kubernetes):
- Levanta **dos pods/servicios aislados** de forma paralela.
- El servicio `telco360-api` expone el puerto para consumo REST.
- El servicio `telco360-ui` arranca Streamlit pasándole dinámicamente la URL interna/externa de la API a través de la variable de entorno `API_URL`, logrando un **Service Discovery** inmediato.

### Requerimientos de Mantenimiento (Operaciones de TI)
- **Monitoreo:** Vigilar el endpoint `/mlops/health`. Si devuelve `Degraded`, indica que los artefactos `.pkl` están ausentes o corrompidos.
- **Dependencias:** Mantener sincronizado `requirements.txt`. Las librerías críticas son `fastapi`, `streamlit`, `scikit-learn==1.3.0` (o superior) y `pandas`. (Nota: Mantener la versión de `scikit-learn` en sincronía entre el entorno de entrenamiento y el de inferencia para evitar incompatibilidades de unpickling).
- **Escalabilidad:** El backend en FastAPI (Uvicorn) es completamente *stateless*, pudiendo ser replicado horizontalmente con un balanceador de carga. El frontend en Streamlit maneja sesión por usuario, por lo que requeriría Sticky Sessions en un despliegue balanceado.
