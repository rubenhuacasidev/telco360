import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Telco Segmentation & Recommendation Engine", version="2.1")

# Resolver paths relativos a la ubicación del script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Cargar artefactos No Supervisados
try:
    with open(os.path.join(MODELS_DIR, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'kmeans_model.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    # Cargar base de datos de clientes leales (para KNN)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    df_loyal = pd.read_csv(os.path.join(DATA_DIR, 'loyal_customers.csv'))
    with open(os.path.join(MODELS_DIR, 'loyal_vectors.pkl'), 'rb') as f:
        loyal_vectors = pickle.load(f)
except Exception as e:
    print(f"Warning: Models not fully loaded. Run the build script first. {e}")

class ClienteInput(BaseModel):
    gender: str = Field(..., description="Female or Male")
    SeniorCitizen: int = Field(..., description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, No")
    OnlineSecurity: str = Field(..., description="Yes, No, No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, No internet service")
    TechSupport: str = Field(..., description="Yes, No, No internet service")
    StreamingTV: str = Field(..., description="Yes, No, No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

def extract_features(cliente: ClienteInput):
    df = pd.DataFrame([cliente.model_dump()])
    # Manejar missing TotalCharges si la antigüedad es 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return preprocessor.transform(df)

@app.post("/segment")
def segment_customer(cliente: ClienteInput):
    """
    Asigna un cliente a un grupo poblacional (Clúster).
    """
    try:
        X_cliente = extract_features(cliente)
        cluster_id = int(kmeans.predict(X_cliente)[0])
        
        perfiles = {
            0: "AHORRADORES. Tienen pocos servicios y son sensibles al precio.<br><b>Táctica Comercial:</b> No intentes venderles paquetes premium. Ofréceles empaquetar su servicio actual en un contrato largo a cambio de un pequeño descuento.",
            1: "VIP (HEAVY USERS). Consumen fibra óptica y múltiples servicios.<br><b>Táctica Comercial:</b> Tienen alto presupuesto. Ofréceles el nivel más alto de Seguridad, Backup o Soporte Técnico. Valoran la calidad sobre el precio.",
            2: "LEALES TRADICIONALES. Alta antigüedad pero contratos básicos.<br><b>Táctica Comercial:</b> Ofréceles 'actualizar' sus equipos o añadir un servicio básico si firman un contrato a largo plazo. Oportunidad para ventas cruzadas suaves.",
            3: "EN RIESGO DE FUGA. Inestabilidad y alto riesgo de cancelación.<br><b>Táctica Comercial:</b> ¡PRIORIDAD MÁXIMA! No intentes sacarles más dinero hoy. Ofréceles un descuento inmediato a cambio de pasar su contrato Mensual a 1 Año para retenerlos."
        }
        
        return {
            "cluster_id": cluster_id,
            "perfil": perfiles.get(cluster_id, "Perfil Desconocido")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def recommend_services(cliente: ClienteInput):
    """
    Motor de Recomendación basado en KNN (Filtrado Colaborativo).
    Busca a los clientes leales más similares dentro del mismo clúster.
    """
    try:
        X_cliente = extract_features(cliente)
        cluster_id = int(kmeans.predict(X_cliente)[0])
        
        # Filtrar leales por el mismo clúster
        mask = df_loyal['Cluster'] == cluster_id
        if not mask.any():
            return {"servicios_recomendados": ["No data"], "justificacion": "No hay suficientes clientes leales en este segmento."}
            
        idx_in_cluster = np.where(mask)[0]
        vectors_in_cluster = loyal_vectors[idx_in_cluster]
        
        # Calcular distancias euclidianas a los clientes de ese clúster
        distancias = np.linalg.norm(vectors_in_cluster - X_cliente, axis=1)
        
        # Encontrar los 5 más cercanos (K=5)
        k = min(5, len(distancias))
        nearest_indices_local = np.argsort(distancias)[:k]
        
        # Índices globales en el df_loyal
        nearest_indices_global = idx_in_cluster[nearest_indices_local]
        vecinos = df_loyal.iloc[nearest_indices_global]
        
        # Analizar qué servicios tienen ellos que nuestro cliente NO tiene
        servicios_posibles = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        precios_estimados = {'OnlineSecurity': 12.0, 'OnlineBackup': 15.0, 'DeviceProtection': 10.0, 'TechSupport': 12.0}
        recomendaciones = []
        impacto_financiero = 0.0
        
        cliente_dict = cliente.model_dump()
        for s in servicios_posibles:
            if cliente_dict[s] == 'No':
                if (vecinos[s] == 'Yes').sum() >= (k / 2):
                    recomendaciones.append(s)
                    impacto_financiero += precios_estimados[s]
                    
        # Regla de Negocio ESTRICTA sobre el modelo KNN
        # Si el cliente tiene contrato mensual, la principal meta es retenerlo con un upgrade, sin importar los vecinos
        if cliente_dict['Contract'] == 'Month-to-month':
            if "Upgrade a Contrato de 1 Año (Fidelización)" not in recomendaciones:
                recomendaciones.append("Upgrade a Contrato de 1 Año (Fidelización)")
                impacto_financiero += 5.0 # Estimación conservadora por retención asegurada
                
        if not recomendaciones:
            return {
                "servicios_recomendados": ["Descuento de Lealtad (Retención Directa)"],
                "justificacion": "Estadísticamente este cliente ya posee los servicios estables de su segmento. No intentes ventas cruzadas hoy. La mejor acción es ofrecer un descuento directo o beneficio exclusivo para asegurar que renueve su suscripción.",
                "impacto_mensual_estimado": 0.0
            }
            
        return {
            "servicios_recomendados": recomendaciones,
            "justificacion": f"Basado en un análisis profundo de {k} perfiles de clientes leales con características idénticas, vender estos servicios cierra la brecha de retención y disminuye la probabilidad de cancelación.",
            "impacto_mensual_estimado": impacto_financiero
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlops/health")
def mlops_health():
    """
    Endpoint para el dashboard de mantenimiento continuo.
    Devuelve la metadatos del modelo actual en producción.
    """
    model_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    if os.path.exists(model_path):
        mod_time = os.path.getmtime(model_path)
        last_trained = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
        size_kb = round(os.path.getsize(model_path) / 1024, 2)
        status = "Healthy"
    else:
        last_trained = "Desconocido"
        size_kb = 0
        status = "Degraded (Model missing)"
        
    return {
        "status": status,
        "model_version": "v2.1 (Unsupervised)",
        "last_retrained": last_trained,
        "algorithm": "K-Means (k=4)",
        "model_size_kb": size_kb
    }
