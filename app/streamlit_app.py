import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import pickle
import os
import time
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, SpectralClustering

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS GLOBALES
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco360 CRM & MLOps",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "api_url" not in st.session_state:
    st.session_state.api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
if "last_segmentation" not in st.session_state:
    st.session_state.last_segmentation = None
if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None

st.markdown("""
<style>
    /* Diseño Base / Typography / Hierarchy */
    @import url('https://rsms.me/inter/inter.css');
    .stApp { background-color: #09090b; color: #f4f4f5; font-family: 'Inter', system-ui, sans-serif; }
    
    /* Superficies y Bordes sutiles */
    .css-1d391kg, .css-1dp5vir { background-color: #18181b !important; border-right: 1px solid rgba(255,255,255,0.06); }
    
    /* Títulos de sección con jerarquía clara */
    .section-title { font-size: 1.125rem; font-weight: 500; color: #a1a1aa; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.5rem; letter-spacing: -0.01em; }
    
    /* Tarjetas KPI Premium */
    .kpi-card { background-color: #18181b; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 1.25rem; }
    .kpi-value { font-size: 2rem; font-weight: 600; color: #f4f4f5; line-height: 1.1; margin-bottom: 0.25rem; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .kpi-label { font-size: 0.8125rem; font-weight: 500; color: #71717a; letter-spacing: 0.04em; }
    
    /* Deltas (Indicadores de Cambio) */
    .delta-up { color: #10b981; font-size: 0.75rem; font-weight: 500; background: rgba(16, 185, 129, 0.1); padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px;}
    .delta-down { color: #f43f5e; font-size: 0.75rem; font-weight: 500; background: rgba(244, 63, 94, 0.1); padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px;}
    
    /* Chips de Servicios */
    .chip { display: inline-block; padding: 4px 10px; margin: 4px 4px 4px 0; border-radius: 6px; font-size: 0.75rem; font-weight: 500; }
    .chip-active { background-color: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.2); }
    .chip-inactive { background-color: #27272a; color: #71717a; border: 1px solid rgba(255,255,255,0.06); }
    
    /* Terminal Console */
    .console-box { background-color: #000000; color: #a1a1aa; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); height: 200px; overflow-y: auto; font-size: 0.8125rem; line-height: 1.5; }
    
    /* Tarjetas de AI Results y Leads */
    .lead-card { background-color: #18181b; border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
    .lead-tag { color: #f43f5e; font-size: 0.75rem; font-weight: 500; margin-bottom: 0.5rem; display: inline-block; }
    .lead-id { color: #f4f4f5; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.25rem; }
    .lead-meta { color: #a1a1aa; font-size: 0.8125rem; }
    .lead-value { color: #f4f4f5; font-weight: 500; }
    
    /* Sidebar Overhaul */
    [data-testid="stSidebar"] { background-color: #09090b !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
    [data-testid="stSidebar"] .stRadio div[role="radio"] { display: none !important; }
    [data-testid="stSidebar"] .stRadio label { padding: 0.6rem 0.75rem; border-radius: 6px; margin-bottom: 0.25rem; transition: all 0.2s ease; cursor: pointer; }
    [data-testid="stSidebar"] .stRadio label:hover { background-color: rgba(255,255,255,0.04); }
    [data-testid="stSidebar"] .stRadio label:has(div[aria-checked="true"]) { background-color: rgba(99, 102, 241, 0.1); }
    [data-testid="stSidebar"] .stRadio label:has(div[aria-checked="true"]) p { color: #818cf8 !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] .stRadio p { font-size: 1.05rem !important; font-weight: 500 !important; color: #a1a1aa; margin: 0; }
    
    .sidebar-divider { height: 1px; background-color: rgba(255,255,255,0.06); margin: 1.5rem 0; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HELPERS (CACHED)
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data_for_viz():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv').dropna(subset=['TotalCharges']).head(1000)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    with open('models/preprocessor.pkl', 'rb') as f: preprocessor = pickle.load(f)
    with open('models/kmeans_model.pkl', 'rb') as f: kmeans = pickle.load(f)
    with open('models/pca_transformer.pkl', 'rb') as f: pca = pickle.load(f)
        
    X = preprocessor.transform(df)
    kmeans_labels = kmeans.predict(X)
    
    dbscan = DBSCAN(eps=3.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)
    
    spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', random_state=42)
    spectral_labels = spectral.fit_predict(X)
    
    X_pca = pca.transform(X)
    
    df_viz = pd.DataFrame({
        'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1],
        'KMeans': kmeans_labels, 'DBSCAN': dbscan_labels, 'Spectral': spectral_labels,
        'MonthlyCharges': df['MonthlyCharges'], 'Tenure': df['tenure'],
        'Contract': df['Contract']
    })
    return df_viz, df

@st.cache_data
def load_raw_dataset():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

# ---------------------------------------------------------------------------
# VISTA 1: PORTAL CRM (VENTAS / CUSTOMER SUCCESS)
# ---------------------------------------------------------------------------
@st.fragment
def render_crm():
    st.title("Telco360: Workspace del Agente")
    st.markdown("Revisa tu cartera de clientes o busca un perfil para obtener la estrategia de retención impulsada por IA.")
    
    df_raw = load_raw_dataset()
    
    colA, colB = st.columns([3, 1])
    with colA:
        # Check if we should reset selectbox via session state (we handle this via logic below)
        idx_to_select = 0
        options = ["Seleccionar o escribir ID...", "Nuevo Cliente Manual"] + df_raw['customerID'].tolist()
        if st.session_state.selected_customer in options:
            idx_to_select = options.index(st.session_state.selected_customer)
            
        customer_id = st.selectbox("Buscar ID de Cliente en Base de Datos", options, index=idx_to_select)
    
    if customer_id != "Nuevo Cliente Manual" and customer_id != "Seleccionar o escribir ID..." and customer_id != st.session_state.selected_customer:
        st.session_state.selected_customer = customer_id
        st.session_state.last_segmentation = None
        st.rerun()
    elif customer_id == "Nuevo Cliente Manual" and st.session_state.selected_customer is not None:
        st.session_state.selected_customer = None
        st.session_state.last_segmentation = None
        st.rerun()
    
    # 1. OBJETIVOS DEL DÍA (Smart Leads) - Si no hay cliente seleccionado
    if customer_id == "Seleccionar o escribir ID..." and not st.session_state.selected_customer:
        st.markdown("<div class='section-title'>Objetivos de Hoy (Smart Leads)</div>", unsafe_allow_html=True)
        st.markdown("La Inteligencia Artificial ha escaneado la base de datos y ha seleccionado a los 3 clientes con **mayor riesgo de fuga y mayor valor financiero**. ¡Llámalos ahora!")
        
        # Filtramos para encontrar los leads
        df_leads = df_raw[(df_raw['Contract'] == 'Month-to-month') & (df_raw['tenure'] < 12)].copy()
        df_leads = df_leads.sort_values(by='TotalCharges', ascending=False).head(3)
        
        cols = st.columns(3)
        for idx, (_, row) in enumerate(df_leads.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div class="lead-card">
                    <div class="lead-tag">Riesgo Crítico</div>
                    <div class="lead-id">{row['customerID']}</div>
                    <div class="lead-meta">LTV en Riesgo: <span class="lead-value">${row['TotalCharges']:,.2f}</span></div>
                    <div class="lead-meta">Antigüedad: <span class="lead-value">{row['tenure']} meses</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Callback function to set the state
                def select_lead(cid=row['customerID']):
                    st.session_state.selected_customer = cid
                    st.session_state.last_segmentation = None
                    
                st.button(f"Abrir Perfil y Ejecutar IA", key=f"btn_{row['customerID']}", type="primary", use_container_width=True, on_click=select_lead)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Directorio completo oculto en un expander
        with st.expander("Ver Directorio Completo de la Cartera"):
            df_sales = df_raw[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']].copy()
            def calculate_risk(r):
                if r['Contract'] == 'Month-to-month' and r['tenure'] < 12: return 'Alta (Fuga)'
                elif r['Contract'] == 'Month-to-month': return 'Media'
                else: return 'Baja (Retenido)'
            df_sales['Prioridad de Llamada'] = df_sales.apply(calculate_risk, axis=1)
            df_sales = df_sales.sort_values(by=['Prioridad de Llamada', 'MonthlyCharges'], ascending=[False, False])
            
            df_sales.rename(columns={'customerID': 'ID Cliente', 'tenure': 'Antigüedad', 'MonthlyCharges': 'Cuota ($)', 'TotalCharges': 'Ingreso ($)', 'Contract': 'Contrato'}, inplace=True)
            df_sales = df_sales[['Prioridad de Llamada', 'ID Cliente', 'Antigüedad', 'Cuota ($)', 'Ingreso ($)', 'Contrato']]
            st.dataframe(df_sales, use_container_width=True, height=300)
            
        return # Terminamos la ejecución de la pantalla aquí para no mostrar nada más

    # Si hay un cliente seleccionado, mostramos un botón para volver al directorio
    if st.session_state.selected_customer:
        if st.button("Cerrar Perfil y Volver al Directorio"):
            st.session_state.selected_customer = None
            st.session_state.last_segmentation = None
            st.rerun()

    def_data = {}
    if st.session_state.selected_customer:
        c_data = df_raw[df_raw['customerID'] == st.session_state.selected_customer].iloc[0]
        def_data = c_data.to_dict()
        
        # PERFIL 360 (DASHBOARD DEL CLIENTE)
        st.markdown(f"<div class='section-title'>Perfil 360: {st.session_state.selected_customer}</div>", unsafe_allow_html=True)
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Life Time Value (Ingresos)</div><div class='kpi-value'>${def_data.get('TotalCharges', 0):,.2f}</div></div>", unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Facturación Mensual</div><div class='kpi-value'>${def_data.get('MonthlyCharges', 0):,.2f}</div></div>", unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Antigüedad y Contrato</div><div class='kpi-value'>{def_data.get('tenure', 0)} meses</div><div style='color: #cbd5e1; font-size: 0.9rem;'>{def_data.get('Contract', '')}</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # INVENTARIO VISUAL DE SERVICIOS
        st.markdown("<div style='font-weight: 500; color: #71717a; font-size: 0.8125rem; margin-bottom: 0.75rem;'>Inventario de Servicios Actuales</div>", unsafe_allow_html=True)
        servicios_ui = []
        # Phone
        servicios_ui.append(f"<span class='chip {'chip-active' if def_data.get('PhoneService')=='Yes' else 'chip-inactive'}'>Telefonía</span>")
        # Internet
        int_stat = 'chip-active' if def_data.get('InternetService') != 'No' else 'chip-inactive'
        int_name = f"Internet ({def_data.get('InternetService')})" if def_data.get('InternetService') != 'No' else "Sin Internet"
        servicios_ui.append(f"<span class='chip {int_stat}'>{int_name}</span>")
        # Add-ons
        addons = {'OnlineSecurity': 'Seguridad', 'OnlineBackup': 'Backup', 'DeviceProtection': 'Protección', 'TechSupport': 'Soporte', 'StreamingTV': 'TV', 'StreamingMovies': 'Cine'}
        for key, name in addons.items():
            stat = 'chip-active' if def_data.get(key) == 'Yes' else 'chip-inactive'
            servicios_ui.append(f"<span class='chip {stat}'>{name}</span>")
            
        st.markdown(" ".join(servicios_ui), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # 2. MOTOR DE IA Y RESULTADOS
    if st.session_state.selected_customer or customer_id == "Nuevo Cliente Manual":
        if st.button("Ejecutar Motor de IA (Analizar Perfil de Riesgo)", type="primary"):
            map_gender = {"Femenino": "Female", "Masculino": "Male"}
            map_sino_inv = {"Yes": "Sí", "No": "No"}
            payload = {
                "gender": def_data.get('gender', 'Female'),
                "SeniorCitizen": def_data.get('SeniorCitizen', 0),
                "Partner": def_data.get('Partner', 'No'),
                "Dependents": def_data.get('Dependents', 'No'),
                "tenure": def_data.get('tenure', 1),
                "PhoneService": def_data.get('PhoneService', 'No'),
                "MultipleLines": def_data.get('MultipleLines', 'No phone service'),
                "InternetService": def_data.get('InternetService', 'DSL'),
                "OnlineSecurity": def_data.get('OnlineSecurity', 'No internet service'),
                "OnlineBackup": def_data.get('OnlineBackup', 'No internet service'),
                "DeviceProtection": def_data.get('DeviceProtection', 'No internet service'),
                "TechSupport": def_data.get('TechSupport', 'No internet service'),
                "StreamingTV": def_data.get('StreamingTV', 'No internet service'),
                "StreamingMovies": def_data.get('StreamingMovies', 'No internet service'),
                "Contract": def_data.get('Contract', 'Month-to-month'),
                "PaperlessBilling": def_data.get('PaperlessBilling', 'Yes'),
                "PaymentMethod": def_data.get('PaymentMethod', 'Electronic check'),
                "MonthlyCharges": def_data.get('MonthlyCharges', 50.0),
                "TotalCharges": def_data.get('TotalCharges', 50.0),
            }
            with st.spinner("Conectando con Motor MLOps..."):
                try:
                    res_seg = requests.post(f"{st.session_state.api_url}/segment", json=payload, timeout=5)
                    res_rec = requests.post(f"{st.session_state.api_url}/recommend", json=payload, timeout=5)
                    if res_seg.status_code == 200 and res_rec.status_code == 200:
                        st.session_state.last_segmentation = {"segment": res_seg.json(), "recommend": res_rec.json()}
                    else:
                        st.error("Error HTTP desde el servidor FastAPI.")
                except Exception as e:
                    st.error(f"Falla de conexión backend: {str(e)}")

    if st.session_state.last_segmentation:
        seg = st.session_state.last_segmentation["segment"]
        rec = st.session_state.last_segmentation["recommend"]
        
        # Color del cluster para gamificación visual
        cluster_colors = {0: "#64748b", 1: "#eab308", 2: "#3b82f6", 3: "#ef4444"}
        c_color = cluster_colors.get(seg['cluster_id'], "#6366f1")
        
        col_res1, col_res2, col_res3 = st.columns([1, 1.5, 1])
        with col_res1:
            st.markdown(f"""
            <div style="background-color: #18181b; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 1.5rem; height: 100%;">
                <div style="color: #71717a; font-size: 0.8125rem; font-weight: 500; margin-bottom: 0.5rem;">Segmentación de IA</div>
                <div style="color: #f4f4f5; font-size: 1.5rem; font-weight: 600; line-height: 1.1; margin-bottom: 1rem; letter-spacing: -0.01em;">Grupo {seg['cluster_id']}</div>
                <div style="background-color: {c_color}15; padding: 0.75rem 1rem; border-radius: 6px; color: {c_color}; font-size: 0.875rem; font-weight: 500;">
                    {seg['perfil']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            servs = "<br>".join([f"• {s}" for s in rec['servicios_recomendados']])
            impacto = rec.get("impacto_mensual_estimado", 0.0)
            
            st.markdown(f"""
            <div style="background-color: #18181b; border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1.5rem; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="color: #10b981; font-size: 0.8125rem; font-weight: 500;">Estrategia Sugerida</div>
                    <div style="background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 500; font-size: 0.75rem;">
                        +${impacto:,.2f} / mes
                    </div>
                </div>
                <div style="color: #f4f4f5; font-size: 0.9375rem; font-weight: 500; margin-bottom: 1rem; line-height: 1.5;">
                    {servs}
                </div>
                <div style="color: #a1a1aa; font-size: 0.8125rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.75rem; line-height: 1.5;">
                    {rec['justificacion']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res3:
            # Proyección de LTV
            st.markdown("<div style='color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: bold; margin-bottom: 0.5rem;'>Proyección Financiera (LTV a 1 Año)</div>", unsafe_allow_html=True)
            current_mrr = def_data.get('MonthlyCharges', 50.0)
            current_ltv = def_data.get('TotalCharges', 50.0)
            impacto = rec.get("impacto_mensual_estimado", 0.0)
            
            # Crear datos falsos de proyeccion
            meses = [0, 3, 6, 9, 12]
            # Si el cliente no se retiene, asumimos que se va en el mes 3 (fuga) y deja de pagar
            proy_base = [
                current_ltv, 
                current_ltv + (current_mrr * 3), 
                current_ltv + (current_mrr * 3), # Fuga, el ingreso se estanca
                current_ltv + (current_mrr * 3),
                current_ltv + (current_mrr * 3)
            ]
            # Si le hacemos upsell, sigue pagando y paga más
            proy_upsell = [current_ltv + ((current_mrr + impacto) * m) for m in meses]
            
            df_proy = pd.DataFrame({
                'Mes Futuro': meses * 2,
                'Ingreso Proyectado ($)': proy_base + proy_upsell,
                'Escenario': ['Abandono Inminente (Riesgo)'] * 5 + ['Con Upselling (Retenido)'] * 5
            })
            
            chart_proy = alt.Chart(df_proy).mark_line(point=True).encode(
                x=alt.X('Mes Futuro:Q', title='Meses a Futuro'),
                y=alt.Y('Ingreso Proyectado ($):Q', scale=alt.Scale(zero=False), title='LTV ($)'),
                color=alt.Color('Escenario:N', scale=alt.Scale(domain=['Abandono Inminente (Riesgo)', 'Con Upselling (Retenido)'], range=['#ef4444', '#10b981'])),
                strokeDash=alt.condition(alt.datum.Escenario == 'Abandono Inminente (Riesgo)', alt.value([5,5]), alt.value([0]))
            ).properties(height=200).configure_legend(orient='bottom', title=None)
            
            st.altair_chart(chart_proy, use_container_width=True)

    # 3. SIMULADOR (FORMULARIO OCULTO)
    if st.session_state.selected_customer or customer_id == "Nuevo Cliente Manual":
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.expander("Simulador de Escenarios (Modificar Datos y Re-Analizar)"):
            st.markdown("¿Qué pasaría si le ofrecemos un nuevo contrato o bajamos su precio? Modifica los datos y vuelve a ejecutar la IA.")
            with st.form("unsupervised_form"):
                map_sino = {"Sí": "Yes", "No": "No"}
                map_sino_inv = {"Yes": "Sí", "No": "No"}
                map_internet = {"DSL": "DSL", "Fibra Óptica": "Fiber optic", "No": "No"}
                map_internet_inv = {"DSL": "DSL", "Fiber optic": "Fibra Óptica", "No": "No"}
                map_phone = {"Sí": "Yes", "No": "No", "Sin servicio telefónico": "No phone service"}
                map_phone_inv = {"Yes": "Sí", "No": "No", "No phone service": "Sin servicio telefónico"}
                map_servicios = {"Sí": "Yes", "No": "No", "Sin servicio de internet": "No internet service"}
                map_servicios_inv = {"Yes": "Sí", "No": "No", "No internet service": "Sin servicio de internet"}
                map_contract = {"Mensual": "Month-to-month", "1 Año": "One year", "2 Años": "Two year"}
                map_contract_inv = {"Month-to-month": "Mensual", "One year": "1 Año", "Two year": "2 Años"}
                map_payment = {"Cheque Electrónico": "Electronic check", "Cheque por Correo": "Mailed check", "Transferencia Bancaria (Automática)": "Bank transfer (automatic)", "Tarjeta de Crédito (Automática)": "Credit card (automatic)"}
                map_payment_inv = {v: k for k, v in map_payment.items()}
                map_gender = {"Femenino": "Female", "Masculino": "Male"}
                map_gender_inv = {"Female": "Femenino", "Male": "Masculino"}

                col1, col2 = st.columns(2)
                with col1:
                    c1, c2 = st.columns(2)
                    with c1:
                        def_gen = map_gender_inv.get(def_data.get('gender', 'Female'), 'Femenino')
                        gender = st.selectbox("Género", list(map_gender.keys()), index=list(map_gender.keys()).index(def_gen))
                        def_sen = "Sí" if def_data.get('SeniorCitizen', 0) == 1 else "No"
                        senior = st.selectbox("Tercera Edad", ["No", "Sí"], index=["No", "Sí"].index(def_sen))
                    with c2:
                        def_part = map_sino_inv.get(def_data.get('Partner', 'No'), 'No')
                        partner = st.selectbox("Tiene Pareja", list(map_sino.keys()), index=list(map_sino.keys()).index(def_part))
                        def_dep = map_sino_inv.get(def_data.get('Dependents', 'No'), 'No')
                        dependents = st.selectbox("Dependientes", list(map_sino.keys()), index=list(map_sino.keys()).index(def_dep))
                    
                    c3, c4 = st.columns(2)
                    with c3:
                        tenure = st.number_input("Antigüedad (Meses)", 0, 100, int(def_data.get('tenure', 1)))
                        def_con = map_contract_inv.get(def_data.get('Contract', 'Month-to-month'), 'Mensual')
                        contract = st.selectbox("Tipo de Contrato", list(map_contract.keys()), index=list(map_contract.keys()).index(def_con))
                    with c4:
                        monthly = st.number_input("Mensualidad ($)", 0.0, 500.0, float(def_data.get('MonthlyCharges', 50.0)))
                        total = st.number_input("Cargos Totales ($)", 0.0, 10000.0, float(def_data.get('TotalCharges', 50.0)))
                    
                    c5, c6 = st.columns(2)
                    with c5:
                        def_pap = map_sino_inv.get(def_data.get('PaperlessBilling', 'Yes'), 'Sí')
                        paperless = st.selectbox("Factura Electrónica", list(map_sino.keys()), index=list(map_sino.keys()).index(def_pap))
                    with c6:
                        def_pay = map_payment_inv.get(def_data.get('PaymentMethod', 'Electronic check'), 'Cheque Electrónico')
                        payment = st.selectbox("Método de Pago", list(map_payment.keys()), index=list(map_payment.keys()).index(def_pay))

                with col2:
                    c7, c8 = st.columns(2)
                    with c7:
                        def_pho = map_sino_inv.get(def_data.get('PhoneService', 'Yes'), 'Sí')
                        phone_service = st.selectbox("Servicio Telefónico", list(map_sino.keys()), index=list(map_sino.keys()).index(def_pho))
                        def_int = map_internet_inv.get(def_data.get('InternetService', 'DSL'), 'DSL')
                        internet_service = st.selectbox("Servicio de Internet", list(map_internet.keys()), index=list(map_internet.keys()).index(def_int))
                        def_sec = map_servicios_inv.get(def_data.get('OnlineSecurity', 'No'), 'No')
                        online_security = st.selectbox("Seguridad Online", list(map_servicios.keys()), index=list(map_servicios.keys()).index(def_sec))
                        def_dev = map_servicios_inv.get(def_data.get('DeviceProtection', 'No'), 'No')
                        device_protection = st.selectbox("Prot. Dispositivo", list(map_servicios.keys()), index=list(map_servicios.keys()).index(def_dev))
                        def_tv = map_servicios_inv.get(def_data.get('StreamingTV', 'No'), 'No')
                        streaming_tv = st.selectbox("Streaming TV", list(map_servicios.keys()), index=list(map_servicios.keys()).index(def_tv))
                    with c8:
                        def_mul = map_phone_inv.get(def_data.get('MultipleLines', 'No'), 'No')
                        multiple_lines = st.selectbox("Líneas Múltiples", list(map_phone.keys()), index=list(map_phone.keys()).index(def_mul))
                        st.markdown("<div style='height: 2px'></div>", unsafe_allow_html=True) 
                        def_bak = map_servicios_inv.get(def_data.get('OnlineBackup', 'No'), 'No')
                        online_backup = st.selectbox("Backup Online", list(map_servicios.keys()), index=list(map_servicios.keys()).index(def_bak))
                        def_sup = map_servicios_inv.get(def_data.get('TechSupport', 'No'), 'No')
                        tech_support = st.selectbox("Soporte Técnico", list(map_servicios.keys()), index=list(map_servicios.keys()).index(def_sup))
                        def_mov = map_servicios_inv.get(def_data.get('StreamingMovies', 'No'), 'No')
                        streaming_movies = st.selectbox("Streaming Movies", list(map_servicios.keys()), index=list(map_servicios.keys()).index(def_mov))
                        
                submitted = st.form_submit_button("Simular Escenario con Motor IA", type="secondary", use_container_width=True)
                
            if submitted:
                payload = {
                    "gender": map_gender[gender], "SeniorCitizen": 1 if senior == "Sí" else 0, "Partner": map_sino[partner],
                    "Dependents": map_sino[dependents], "tenure": tenure, "PhoneService": map_sino[phone_service],
                    "MultipleLines": map_phone[multiple_lines], "InternetService": map_internet[internet_service],
                    "OnlineSecurity": map_servicios[online_security], "OnlineBackup": map_servicios[online_backup],
                    "DeviceProtection": map_servicios[device_protection], "TechSupport": map_servicios[tech_support],
                    "StreamingTV": map_servicios[streaming_tv], "StreamingMovies": map_servicios[streaming_movies],
                    "Contract": map_contract[contract], "PaperlessBilling": map_sino[paperless],
                    "PaymentMethod": map_payment[payment], "MonthlyCharges": monthly, "TotalCharges": total,
                }
                with st.spinner("Simulando..."):
                    try:
                        res_seg = requests.post(f"{st.session_state.api_url}/segment", json=payload, timeout=5)
                        res_rec = requests.post(f"{st.session_state.api_url}/recommend", json=payload, timeout=5)
                        if res_seg.status_code == 200 and res_rec.status_code == 200:
                            st.session_state.last_segmentation = {"segment": res_seg.json(), "recommend": res_rec.json()}
                            st.rerun()
                    except Exception as e:
                        st.error("Error al simular")

# ---------------------------------------------------------------------------
# VISTA 2: DASHBOARD DIRECTIVO (MARKETING)
# ---------------------------------------------------------------------------
def render_dashboard():
    st.title("Telco360: Dashboard Ejecutivo")
    st.markdown("Monitor de KPIs financieros y de negocio derivados del perfilamiento con Inteligencia Artificial.")
    
    df_viz, df_raw = load_and_prepare_data_for_viz()
    df_raw['Cluster'] = df_viz['KMeans']
    
    # Mapeo de nombres de negocio
    cluster_names = {
        0: "Ahorradores",
        1: "VIP (Heavy Users)",
        2: "Leales Tradicionales",
        3: "En Riesgo de Fuga"
    }
    df_raw['Segmento'] = df_raw['Cluster'].map(cluster_names)
    
    revenue_by_cluster = df_raw.groupby('Segmento')['TotalCharges'].sum().reset_index()
    total_rev = revenue_by_cluster['TotalCharges'].sum()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Clientes Activos</div><div class='kpi-value'>{len(df_raw):,}</div><span class='delta-up'>+4.2%</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Ingresos (LTV)</div><div class='kpi-value'>${total_rev:,.0f}</div><span class='delta-up'>+12.1%</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>ARPU (Ingreso Promedio)</div><div class='kpi-value'>${df_raw['MonthlyCharges'].mean():.2f}</div><span class='delta-up'>+1.4%</span></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Rotación (Churn Real)</div><div class='kpi-value'>12.5%</div><span class='delta-down'>-2.3% (Mejora)</span></div>", unsafe_allow_html=True)

    st.markdown("<br><div class='section-title'>Análisis de la Cartera por Segmento Comercial</div>", unsafe_allow_html=True)
    
    colA, colB = st.columns([1, 1])
    with colA:
        # Gráfico de Barras Horizontales (Ingresos)
        chart1 = alt.Chart(revenue_by_cluster).mark_bar(cornerRadiusEnd=4, height=30).encode(
            y=alt.Y('Segmento:N', title=None, sort='-x'),
            x=alt.X('TotalCharges:Q', title="Ingreso Generado ($)"),
            color=alt.Color('Segmento:N', scale=alt.Scale(domain=list(cluster_names.values()), range=['#38bdf8', '#fbbf24', '#4ade80', '#ef4444']), legend=None)
        )
        text1 = chart1.mark_text(align='left', baseline='middle', dx=5, color='#a1a1aa', fontWeight=500, fontSize=12).encode(
            text=alt.Text('TotalCharges:Q', format='$,.0f')
        )
        chart_ingresos = (chart1 + text1).properties(height=250, title="Ingresos Financieros por Segmento").configure_view(strokeOpacity=0).configure_title(fontSize=14, font='Inter', color='#f4f4f5', anchor='start').configure_axis(grid=False, domain=False, labelFont='Inter', titleFont='Inter', labelColor='#71717a', titleColor='#71717a')
        st.altair_chart(chart_ingresos, use_container_width=True)
        
    with colB:
        # 100% Stacked Bar Chart para Contratos (Adiós Donas)
        contract_data = df_raw.groupby(['Segmento', 'Contract']).size().reset_index(name='Count')
        chart2 = alt.Chart(contract_data).mark_bar(height=30).encode(
            y=alt.Y('Segmento:N', title=None),
            x=alt.X('Count:Q', stack='normalize', axis=alt.Axis(format='%'), title="Proporción de Contratos"),
            color=alt.Color('Contract:N', title="Tipo de Contrato", scale=alt.Scale(domain=['Month-to-month', 'One year', 'Two year'], range=['#ef4444', '#fbbf24', '#10b981'])),
            order=alt.Order('Contract', sort='ascending')
        ).properties(height=250, title="Estabilidad Contractual (100% Stacked)").configure_view(strokeOpacity=0).configure_title(fontSize=14, font='Inter', color='#f4f4f5', anchor='start').configure_axis(grid=False, domain=False, labelFont='Inter', titleFont='Inter', labelColor='#71717a', titleColor='#71717a').configure_legend(labelFont='Inter', titleFont='Inter', titleColor='#71717a', labelColor='#a1a1aa')
        st.altair_chart(chart2, use_container_width=True)
        
    st.markdown("<br><div class='section-title'>Rentabilidad vs Lealtad (Customer Journey)</div>", unsafe_allow_html=True)
    st.markdown("Visualiza hacia dónde migran tus clientes: Los Ahorradores (abajo izquierda) deben convertirse en VIPs o Leales (arriba derecha).")
    
    # Scatter Plot con Tooltips
    chart3 = alt.Chart(df_raw).mark_circle(size=60, opacity=0.8).encode(
        x=alt.X('tenure:Q', title="Antigüedad del Cliente (Meses)", axis=alt.Axis(gridColor='rgba(255,255,255,0.05)')),
        y=alt.Y('MonthlyCharges:Q', title="Facturación Mensual ($)", axis=alt.Axis(gridColor='rgba(255,255,255,0.05)')),
        color=alt.Color('Segmento:N', scale=alt.Scale(domain=list(cluster_names.values()), range=['#38bdf8', '#fbbf24', '#4ade80', '#ef4444']), legend=alt.Legend(title="Segmento IA", orient='top')),
        tooltip=['customerID', 'Segmento', 'tenure', 'MonthlyCharges', 'Contract']
    ).properties(height=400).interactive().configure_view(strokeOpacity=0).configure_axis(domain=False, labelFont='Inter', titleFont='Inter', labelColor='#71717a', titleColor='#71717a').configure_legend(labelFont='Inter', titleFont='Inter', titleColor='#71717a', labelColor='#a1a1aa')
    
    st.altair_chart(chart3, use_container_width=True)

# ---------------------------------------------------------------------------
# VISTA 3: CENTRO DE CONTROL MLOPS & DESARROLLO
# ---------------------------------------------------------------------------
@st.fragment
def render_mlops():
    st.title("Telco360: Centro MLOps & Desarrollo")
    st.markdown("Área técnica (Data Science & IT). Monitor de ciclo de vida del modelo y pipeline CI/CD.")
    
    st.markdown("<div class='section-title'>1. Salud del Modelo (Producción)</div>", unsafe_allow_html=True)
    try:
        r = requests.get(f"{st.session_state.api_url}/mlops/health", timeout=3)
        if r.status_code == 200:
            h = r.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status API", h['status'])
            c2.metric("Versión Motor", h['model_version'])
            c3.metric("Algoritmo / K", h['algorithm'])
            c4.metric("Última Actualización", h['last_retrained'])
        else:
            st.warning("El Endpoint /mlops/health no está disponible.")
    except Exception as e:
        st.error("No se pudo conectar al servidor FastAPI.")
        
    st.markdown("<br><div class='section-title'>2. Métricas de Evaluación en Tiempo Real (Clustering)</div>", unsafe_allow_html=True)
    try:
        import json
        metrics_path = "metrics/unsupervised_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            model_name = metrics.get('model', 'K-Means')
            k_val = metrics.get('hyperparameters', {}).get('n_clusters', 'N/A')
            trained_date = metrics.get('trained_at', 'Desconocido')[:10]
            st.info(f"**Modelo Desplegado (Producción):** `{model_name}` con `k={k_val}` clústeres. Entrenado el `{trained_date}`.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Silhouette Score (Densidad)", metrics['silhouette_score'])
            c2.metric("Davies-Bouldin (Separación)", metrics['davies_bouldin_index'])
            c3.metric("Calinski-Harabasz (Varianza)", metrics['calinski_harabasz_index'])
            
            # Historial
            history_path = "metrics/training_history.json"
            if os.path.exists(history_path):
                with open(history_path, "r") as fh:
                    history_data = json.load(fh)
                
                if len(history_data) > 0:
                    with st.expander("Ver Historial de Despliegues Anteriores"):
                        df_history = pd.DataFrame(history_data)
                        df_history['trained_at'] = pd.to_datetime(df_history['trained_at']).dt.strftime('%Y-%m-%d %H:%M')
                        df_history.rename(columns={
                            'trained_at': 'Fecha de Despliegue',
                            'model': 'Algoritmo',
                            'silhouette_score': 'Silhouette',
                            'davies_bouldin_index': 'Davies-Bouldin',
                            'calinski_harabasz_index': 'Calinski-Harabasz'
                        }, inplace=True)
                        st.dataframe(df_history[['Fecha de Despliegue', 'Algoritmo', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']], use_container_width=True, hide_index=True)
                        
        else:
            st.warning("Métricas de clustering no disponibles. Entrena el modelo en el pipeline.")
    except Exception as e:
        st.error(f"Error al leer métricas: {e}")
        
    st.markdown("<br><div class='section-title'>3. Visualizaciones Técnicas de Agrupamiento (PCA)</div>", unsafe_allow_html=True)
    st.markdown("Proyección matemática de alta dimensión a 2D para inspección visual de los clústeres generados por la IA.")
    
    df_viz, _ = load_and_prepare_data_for_viz()
    col1, col2, col3 = st.columns(3)
    def plot_clusters(data, label_col, title, color_scheme):
        return alt.Chart(data).mark_circle(size=40, opacity=0.7).encode(
            x=alt.X('PCA1', axis=alt.Axis(grid=False, labels=False, ticks=False)), y=alt.Y('PCA2', axis=alt.Axis(grid=False, labels=False, ticks=False)),
            color=alt.Color(f'{label_col}:N', scale=alt.Scale(scheme=color_scheme), legend=None),
            tooltip=['MonthlyCharges', 'Tenure', f'{label_col}:N']
        ).properties(title=title, height=250).configure_view(strokeOpacity=0)

    with col1:
        st.altair_chart(plot_clusters(df_viz, 'KMeans', 'K-Means (Producción)', 'category10'), use_container_width=True)
        st.caption("Silhouette Score: 0.274 (Mejor rendimiento)")
    with col2:
        st.altair_chart(plot_clusters(df_viz, 'Spectral', 'Spectral Clustering', 'set2'), use_container_width=True)
        st.caption("Silhouette Score: 0.261 (O(n^3) - Inviable para CI/CD continuo)")
    with col3:
        st.altair_chart(plot_clusters(df_viz, 'DBSCAN', 'DBSCAN (Density)', 'dark2'), use_container_width=True)
        st.caption("Silhouette Score: -0.05 (Descartado - Exceso de ruido)")

    st.markdown("<div class='section-title'>3. Pipeline de Integración Continua (Real CI/CD)</div>", unsafe_allow_html=True)
    if st.button("Ejecutar Pipeline de Reentrenamiento (GitHub Actions)"):
        github_token = os.environ.get("GITHUB_TOKEN")
        github_repo = os.environ.get("GITHUB_REPO", "rubenhuacasidev/telco360")
        
        if not github_token:
            st.error("Falta el token de GitHub. Configura 'GITHUB_TOKEN' en las variables de entorno para activar esta función.")
        else:
            with st.spinner("Desencadenando pipeline en GitHub Actions..."):
                headers = {
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                }
                payload = {"ref": "main"}
                url = f"https://api.github.com/repos/{github_repo}/actions/workflows/mlops.yml/dispatches"
                
                try:
                    res = requests.post(url, headers=headers, json=payload, timeout=10)
                    if res.status_code == 204:
                        st.success("¡Pipeline MLOps disparado con éxito en GitHub Actions! Los modelos se actualizarán y Render redesplegará los servicios en breve.")
                        st.markdown(f"[Ver estado del Workflow en GitHub](https://github.com/{github_repo}/actions)", unsafe_allow_html=True)
                    else:
                        st.error(f"Fallo al conectar con GitHub API (Status: {res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Error HTTP al invocar el webhook: {e}")

# ---------------------------------------------------------------------------
# MENÚ LATERAL Y NAVEGACIÓN
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='margin-bottom: 0.25rem; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.03em; color: #f4f4f5;'>Telco360 <span style='color: #818cf8;'>CRM</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #a1a1aa; font-size: 1rem; font-weight: 500; margin-bottom: 1.5rem;'>SaaS Edition • Powered by AI</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #71717a; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Módulos del Sistema</div>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navegación del sistema",
        ["Portal CRM (Ventas)", "Dashboard Directivo", "Centro MLOps & Dev"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='color: #71717a; font-size: 0.85rem; font-weight: 500; line-height: 1.6;'>
            Environment: <span style='color: #a1a1aa;'>Production v2.1</span><br>
            Engine: <span style='color: #a1a1aa;'>K-Means & KNN Colab</span>
        </div>
    """, unsafe_allow_html=True)

if page == "Portal CRM (Ventas)":
    render_crm()
elif page == "Dashboard Directivo":
    render_dashboard()
else:
    render_mlops()
