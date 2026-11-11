import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y TEMA INSTITUCIONAL (EXTERNADO DE COLOMBIA)
# ==============================================================================
st.set_page_config(
    page_title="Wi-Fi Indoor Localization | MNAV - Universidad Externado",
    page_icon="📍",
    layout="wide"
)

# Inyección de CSS institucional (Verde Externado #005A36) y estilo mejorado
st.markdown("""
    <style>
        /* Color institucional Universidad Externado */
        :root {
            --primary-color: #005A36;
            --secondary-color: #004227;
            --light-bg: #f4f9f6;
        }
        
        /* Personalización de títulos y banners */
        .main-header {
            background-color: var(--primary-color);
            padding: 20px 25px;
            border-radius: 12px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 {
            color: #ffffff !important;
            margin: 0;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .main-header p {
            color: #d1e7dd !important;
            margin: 5px 0 0 0;
            font-size: 0.95rem;
        }
        
        /* Contenedores de métricas */
        div[data-testid="stMetricValue"] {
            color: var(--primary-color) !important;
            font-weight: 700;
        }
        
        /* Ajuste de botones en sidebar */
        .stButton>button {
            background-color: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: var(--secondary-color) !important;
        }

        /* Formato para el logo */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: white;
            padding: 10px 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-bottom: 3px solid var(--primary-color);
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CABECERA E IDENTIDAD INSTITUCIONAL
# ==============================================================================
col_logo, col_info = st.columns([1, 4])

with col_logo:
    # Intenta cargar la imagen enviada por el usuario si está en el repositorio, o usa el render remoto
    st.image("image_4b1621.png", use_container_width=True)

with col_info:
    st.markdown("""
        <div class="main-header">
            <h1>📍 Sistema de Localización Wi-Fi en Interiores</h1>
            <p>Proyecto de Machine Learning: Algoritmo QA-kNN | Museo Nacional de Artes Visuales (MNAV)</p>
            <p style="font-size: 0.85rem; opacity: 0.9;"><strong>Universidad Externado de Colombia</strong> — Facultad de Economía / Ciencia de Datos</p>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. CARGA DE MODELO Y DATOS (CACHEADO)
# ==============================================================================
@st.cache_resource
def cargar_artefactos():
    model = joblib.load('modelo_knn.pkl')
    scaler = joblib.load('escalador.pkl')
    df = pd.read_csv('ejemplos_huellas.csv')
    
    # Extraer de forma estricta las 188 columnas de APs
    columnas_rssi = [c for c in df.columns if c.startswith('AP_')]
    
    return model, scaler, df, columnas_rssi

try:
    model, scaler, df_crudo, columnas_rssi = cargar_artefactos()
except Exception as e:
    st.error(f"🚨 Error al cargar los artefactos del modelo: {e}")
    st.info("Asegúrate de que 'modelo_knn.pkl', 'escalador.pkl' y 'ejemplos_huellas.csv' existan en la raíz del repositorio.")
    st.stop()

# ==============================================================================
# 4. CONTROLES Y SELECCIÓN EN BARRA LATERAL
# ==============================================================================
st.sidebar.image("image_4b1621.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.header("🕹️ Panel de Experimentación")

modo_entrada = st.sidebar.radio(
    "Selecciona el origen de la huella RSSI:",
    ["Huella de referencia (Por Zona MNAV)", "Generación estocástica (Simulación)", "Cargar archivo CSV"]
)

huella_input = None
zona_real_ejemplo = None

if modo_entrada == "Huella de referencia (Por Zona MNAV)":
    zonas_disponibles = sorted(df_crudo["location"].unique())
    zona_seleccionada = st.sidebar.selectbox("Selecciona la región real de control:", zonas_disponibles)
    
    muestras_zona = df_crudo[df_crudo["location"] == zona_seleccionada]
    idx_sample = st.sidebar.slider("Índice de muestra de la zona:", 0, len(muestras_zona) - 1, 0)
    
    # Extraer estrictamente las 188 columnas en formato matriz (1, 188)
    huella_df = muestras_zona.iloc[[idx_sample]][columnas_rssi]
    huella_input = huella_df.values
    zona_real_ejemplo = zona_seleccionada

elif modo_entrada == "Generación estocástica (Simulación)":
    if st.sidebar.button("🎲 Sintetizar nueva lectura RSSI"):
        huella_input = np.random.uniform(-99, -45, size=(1, len(columnas_rssi)))
    else:
        huella_input = np.random.uniform(-99, -45, size=(1, len(columnas_rssi)))

elif modo_entrada == "Cargar archivo CSV":
    archivo_subido = st.sidebar.file_uploader("Sube un CSV de 188 columnas (AP_001 a AP_188)", type=["csv"])
    if archivo_subido is not None:
        df_subido = pd.read_csv(archivo_subido)
        cols_subidas = [c for c in df_subido.columns if c.startswith('AP_')]
        if len(cols_subidas) == len(columnas_rssi):
            huella_input = df_subido[cols_subidas].iloc[[0]].values
        else:
            st.error(f"El archivo debe incluir exactamente {len(columnas_rssi)} Access Points. Se detectaron {len(cols_subidas)}.")

# ==============================================================================
# 5. PREDICCIÓN Y PRESENTACIÓN DE RESULTADOS
# ==============================================================================
col1, col2 = st.columns([1, 1])

if huella_input is not None:
    # Garantizar compatibilidad estricta de 188 columnas (Resuelve el ValueError)
    if huella_input.shape[1] == len(columnas_rssi):
        
        # Transformación con el StandardScaler preentrenado e Inferencia k-NN
        huella_scaled = scaler.transform(huella_input)
        prediccion_zona = model.predict(huella_scaled)[0]
        probas = model.predict_proba(huella_scaled)[0]
        confianza_max = np.max(probas) * 100
        
        with col1:
            st.subheader("🎯 Clasificación de la Posición")
            
            # Tarjeta destacada de resultado
            st.metric(label="Zona Estimada por el Modelo k-NN", value=str(prediccion_zona))
            
            if zona_real_ejemplo:
                st.info(f"📍 **Ubicación Real de Referencia:** `{zona_real_ejemplo}`")
                if str(prediccion_zona) == str(zona_real_ejemplo):
                    st.success("✅ **Coincidencia Exacta:** El clasificador estimó correctamente la sala.")
                else:
                    st.warning("⚠️ **Desviación:** Estimación en una sala adyacente o con alta similitud de señal.")
            
            st.metric(label="Métrica de Confianza / Probabilidad", value=f"{confianza_max:.1f}%")
            
            # Umbral de advertencia operacional
            if confianza_max < 40:
                st.error("🚨 **ALERTA DE BAJA CONFIANZA:** La huella de señal difiere de los perfiles regionales conocidos. El usuario puede estar fuera de la zona calibrada.")

        with col2:
            st.subheader("📊 Perfil de Potencia RSSI (Top 10 APs más Fuertes)")
            df_huella = pd.DataFrame({
                "Access Point": columnas_rssi,
                "Señal RSSI (dBm)": huella_input[0]
            })
            top_10_aps = df_huella.sort_values(by="Señal RSSI (dBm)", ascending=False).head(10)
            
            # Tabla estilizada
            st.dataframe(top_10_aps, use_container_width=True, hide_index=True)
            
    else:
        st.error(f"❌ Error de dimensión: La matriz de entrada tiene {huella_input.shape[1]} columnas, pero el modelo requiere {len(columnas_rssi)}.")

# ==============================================================================
# 6. ADVERTENCIAS Y LIMITACIONES OPERACIONALES (APARTADO REQUERIDO)
# ==============================================================================
st.markdown("---")
st.subheader("⚠️ Consideraciones Operacionales del Despliegue")

col_adv1, col_adv2, col_adv3 = st.columns(3)

with col_adv1:
    st.markdown("""
    **📱 Heterogeneidad de Hardware**
    Las diferencias en la antena Wi-Fi de cada smartphone introducen sesgos de ganancia en las lecturas de RSSI.
    """)

with col_adv2:
    st.markdown("""
    **🧱 Dinámica del Entorno**
    El flujo de visitantes y modificaciones en tabiques o mobiliario atenúan la propagación del flujo de onda RF.
    """)

with col_adv3:
    st.markdown("""
    **📡 Infraestructura de Red**
    Si un router de la red se desactiva o cambia su ubicación física, se requiere re-calibrar la huella digital regional.
    """)
