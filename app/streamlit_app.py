"""
streamlit_app.py — Interfaz visual para predicción de Churn.
Diseño moderno estilo Dashboard.
"""

import os
import streamlit as st
import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Telco Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CSS Personalizado
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Forzar fondo oscuro en el sidebar y texto blanco */
    [data-testid="stSidebar"] {
        background-color: #1f2937 !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Subtítulos */
    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
        color: #1f2937 !important;
    }
    
    /* Titulos h1, h2 para sobreescribir si esta en dark mode por error */
    h1, h2 {
        color: #1f2937 !important;
    }
    p, label, .stMarkdown {
        color: #374151 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Botón principal (contenido) */
    .main .stButton > button {
        height: 3rem;
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 1rem;
        border: none;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    
    /* Botones del Sidebar (Navegación) */
    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start;
        padding-left: 1rem;
        background-color: transparent;
        border: none;
        color: #e5e7eb !important;
        font-weight: 500;
        height: auto;
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
        margin-top: 0;
        margin-bottom: 0.25rem;
        box-shadow: none;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff !important;
        transform: none;
        box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(59, 130, 246, 0.9) !important;
        color: #ffffff !important;
        border-left: 4px solid #ffffff;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar (Navegación Interactiva Real)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff !important; margin-bottom: 2rem;'>TELCO MLOPS</h2>", unsafe_allow_html=True)
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard General"
        
    def nav_to(page_name):
        st.session_state.current_page = page_name
        
    # Navegación basada en botones
    st.button("Dashboard General", on_click=nav_to, args=("Dashboard General",), use_container_width=True, type="primary" if st.session_state.current_page == "Dashboard General" else "secondary")
    st.button("Predicción Individual", on_click=nav_to, args=("Predicción Individual",), use_container_width=True, type="primary" if st.session_state.current_page == "Predicción Individual" else "secondary")
    st.button("Analítica de Cohortes", on_click=nav_to, args=("Analítica de Cohortes",), use_container_width=True, type="primary" if st.session_state.current_page == "Analítica de Cohortes" else "secondary")
    st.button("Configuración del Modelo", on_click=nav_to, args=("Configuración del Modelo",), use_container_width=True, type="primary" if st.session_state.current_page == "Configuración del Modelo" else "secondary")
    
    page = st.session_state.current_page
    
    st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.caption("v1.0.0 MLOps Pipeline")
    st.caption("Estado de API: 🟢 Conectado")

# ---------------------------------------------------------------------------
# PÁGINA 1: DASHBOARD
# ---------------------------------------------------------------------------
if page == "Dashboard General":
    st.title("Vista General de Negocio")
    st.markdown("Métricas clave de retención y comportamiento de clientes en el último trimestre.")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Clientes Totales", "7,043", "+124 este mes")
    kpi2.metric("Tasa de Churn Global", "26.5%", "-2.1% vs mes anterior", delta_color="inverse")
    kpi3.metric("MRR (Ingreso Recurrente)", "$456.2K", "+$12.5K")
    kpi4.metric("Valor Vida Cliente (LTV)", "$1,205", "+$45")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos simulados
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h3>Retención Histórica vs Predicción (Últimos 6 meses)</h3>", unsafe_allow_html=True)
        # Simular datos para el chart
        dates = pd.date_range(end=pd.Timestamp.today(), periods=6, freq='M')
        data = pd.DataFrame({
            "Retención Real": [85, 82, 80, 81, 79, 78],
            "Retención Predicha": [84, 83, 79, 80, 78, 77]
        }, index=dates)
        st.line_chart(data)
        
    with col2:
        st.markdown("<h3>Distribución de Riesgo</h3>", unsafe_allow_html=True)
        risk_data = pd.DataFrame({
            "Segmento": ["Riesgo Alto", "Riesgo Medio", "Estable"],
            "Volumen": [1850, 2100, 3093]
        })
        st.bar_chart(risk_data.set_index("Segmento"), color="#3b82f6")

# ---------------------------------------------------------------------------
# PÁGINA 2: PREDICCIÓN INDIVIDUAL (El formulario real)
# ---------------------------------------------------------------------------
elif page == "Predicción Individual":
    st.title("Motor Predictivo de Fuga (Churn)")
    st.markdown("Complete el perfil del cliente para estimar la probabilidad de cancelación del servicio.")

    with st.form("prediccion_form", clear_on_submit=False):
        
        st.markdown("<h3>Información Personal</h3>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Género", ["Female", "Male"])
        with c2:
            senior = st.selectbox("Ciudadano Senior", [0, 1], format_func=lambda x: "Sí" if x else "No")
        with c3:
            partner = st.selectbox("Tiene Pareja", ["Yes", "No"])
        with c4:
            dependents = st.selectbox("Tiene Dependientes", ["Yes", "No"])

        st.markdown("<h3>Servicios de Telefonía e Internet</h3>", unsafe_allow_html=True)
        c5, c6, c7 = st.columns(3)
        with c5:
            phone_service = st.selectbox("Servicio Telefónico", ["Yes", "No"])
            multiple_lines = st.selectbox("Líneas Múltiples", ["Yes", "No", "No phone service"])
        with c6:
            internet_service = st.selectbox("Tipo de Conexión", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Seguridad en Línea", ["Yes", "No", "No internet service"])
        with c7:
            device_protection = st.selectbox("Protección de Dispositivo", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Backup en Línea", ["Yes", "No", "No internet service"])

        st.markdown("<h3>Complementos y Entretenimiento</h3>", unsafe_allow_html=True)
        c8, c9, c10 = st.columns(3)
        with c8:
            tech_support = st.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"])
        with c9:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        with c10:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        st.markdown("<h3>Contrato y Facturación</h3>", unsafe_allow_html=True)
        c11, c12, c13, c14 = st.columns(4)
        with c11:
            tenure = st.number_input("Antigüedad (Meses)", min_value=0, max_value=120, value=12)
        with c12:
            contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        with c13:
            paperless = st.selectbox("Facturación Digital", ["Yes", "No"])
        with c14:
            payment = st.selectbox(
                "Método de Pago",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

        c15, c16, c17 = st.columns([1, 1, 2])
        with c15:
            monthly = st.number_input("Cargo Mensual (USD)", min_value=0.0, value=70.0, step=5.0)
        with c16:
            total = st.number_input("Cargo Total (USD)", min_value=0.0, value=840.0, step=50.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Ejecutar Análisis Predictivo", use_container_width=True)

    if submitted:
        payload = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                churn = result["churn"]
                prob = result["probabilidad"]
                
                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.markdown("<h3>Estado del Cliente</h3>", unsafe_allow_html=True)
                    if churn:
                        st.error("Riesgo Crítico de Fuga Detectado")
                    else:
                        st.success("Cliente Estable")
                        
                    st.metric(
                        label="Probabilidad de Abandono",
                        value=f"{prob * 100:.1f}%",
                        delta="Requiere retención" if churn else "Niveles normales",
                        delta_color="inverse"
                    )
                    
                with res_col2:
                    st.markdown("<h3>Nivel de Riesgo</h3>", unsafe_allow_html=True)
                    color = "#ef4444" if churn else "#10b981"
                    st.markdown(
                        f"""
                        <div style="width: 100%; background-color: #e5e7eb; border-radius: 8px; height: 24px; overflow: hidden; margin-top: 15px;">
                            <div style="width: {prob * 100}%; background-color: {color}; height: 100%;"></div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("El modelo procesó las características ingresadas utilizando Regresión Logística (C=10, lbfgs). El umbral de decisión automático clasifica a los clientes con alta probabilidad como riesgo crítico.")

            else:
                st.error(f"Error de comunicación con el motor predictivo: HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("No se pudo establecer conexión con la API predictiva. Verifique que el servicio backend esté activo.")
        except Exception as e:
            st.error(f"Error interno durante la ejecución: {str(e)}")

# ---------------------------------------------------------------------------
# PÁGINA 3: ANALÍTICA DE COHORTES
# ---------------------------------------------------------------------------
elif page == "Analítica de Cohortes":
    st.title("Analítica Predictiva por Cohortes")
    st.markdown("Desglose del riesgo de fuga segmentado por características demográficas y de servicio.")
    
    st.markdown("<h3>Riesgo Promedio por Tipo de Contrato (Tendencia)</h3>", unsafe_allow_html=True)
    # Datos simulados con tendencia
    chart_data = pd.DataFrame(
        np.random.randn(20, 3) * 5 + [40, 15, 5],
        columns=['Month-to-Month', 'One Year', 'Two Year']
    )
    st.area_chart(chart_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3>Top Factores de Fuga</h3>", unsafe_allow_html=True)
        factores = pd.DataFrame({
            "Importancia": [0.85, 0.62, 0.45, 0.31, 0.22],
            "Característica": ["Contrato Mensual", "Fibra Óptica", "Sin Seguridad Online", "Alta Facturación", "Baja Antigüedad"]
        })
        st.dataframe(factores, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("<h3>Distribución Geográfica de Fuga</h3>", unsafe_allow_html=True)
        # Mapa simulado
        map_data = pd.DataFrame(
            np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
            columns=['lat', 'lon'])
        st.map(map_data, zoom=10)

# ---------------------------------------------------------------------------
# PÁGINA 4: CONFIGURACIÓN
# ---------------------------------------------------------------------------
else:
    st.title("Configuración del Sistema MLOps")
    st.markdown("Administre los endpoints y parámetros del modelo de predicción.")
    
    st.markdown("<h3>Parámetros de Conexión</h3>", unsafe_allow_html=True)
    st.text_input("URL Base de la API", value=API_URL)
    st.text_input("API Key (Opcional)", type="password")
    st.button("Probar Conexión Backend")
    
    st.markdown("<h3>Parámetros del Modelo</h3>", unsafe_allow_html=True)
    st.slider("Umbral de Decisión de Fuga (Threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.selectbox("Modelo Activo", ["Regresión Logística (v1.0.0 - Producción)", "Random Forest (v0.9.1 - Deprecated)"])
    st.button("Guardar Configuración", type="primary")

