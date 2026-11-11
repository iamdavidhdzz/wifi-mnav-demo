import pandas as pd
import numpy as np
import streamlit as st
import joblib

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA E IDENTIDAD VISUAL (EXTERNADO)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Wi-Fi Indoor Localization - Universidad Externado",
    page_icon="📍",
    layout="wide"
)

# Estilos CSS personalizados (Verde Institucional #005A36)
st.markdown("""
    <style>
    .main-header {
        color: #005A36;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #4A5568;
        font-size: 1.0rem;
        margin-bottom: 20px;
    }
    div[data-testid="stSidebar"] {
        background-color: #F4F7F5;
        border-right: 1px solid #E0E8E3;
    }
    .stButton>button {
        background-color: #005A36;
        color: white;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. ENCABEZADO Y LOGO INSTITUCIONAL
# ------------------------------------------------------------------------------
col_logo, col_titulo = st.columns([1, 4])

with col_logo:
    try:
        st.image("image_4b1621.png", use_container_width=True)
    except Exception:
        st.markdown("## 🏛️ **EXTERNADO**")

with col_titulo:
    st.markdown('<h1 class="main-header">Localización Wi-Fi en Interiores</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema de estimación de regiones (MNAV) mediante k-NN | Facultad de Economía / Ciencia de Datos</p>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# 3. CARGA DE MODELO Y DATOS (CACHEADO)
# ------------------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    model = joblib.load('modelo_knn.pkl')
    scaler = joblib.load('escalador.pkl')
    df = pd.read_csv('ejemplos_huellas.csv')
    
    columnas_rssi = [c for c in df.columns if c.startswith('AP_')]
    return model, scaler, df, columnas_rssi

try:
    model, scaler, df_crudo, columnas_rssi = cargar_artefactos()
except Exception as e:
    st.error(f"🚨 Error cargando artefactos del modelo: {e}")
    st.stop()

# Determinar características esperadas por el escalador
n_features_esperadas = getattr(scaler, "n_features_in_", len(columnas_rssi))

# ------------------------------------------------------------------------------
# 4. BARRA LATERAL (CONTROLES)
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 🕹️ Panel de Control")

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
    
    huella_df = muestras_zona.iloc[[idx_sample]][columnas_rssi]
    huella_input = huella_df.values
    zona_real_ejemplo = zona_seleccionada

elif modo_entrada == "Generación estocástica (Simulación)":
    if st.sidebar.button("🎲 Generar nueva huella RSSI"):
        huella_input = np.random.uniform(-99, -45, size=(1, len(columnas_rssi)))
    else:
        huella_input = np.random.uniform(-99, -45, size=(1, len(columnas_rssi)))

elif modo_entrada == "Cargar archivo CSV":
    archivo_subido = st.sidebar.file_uploader("Sube un CSV con características AP_", type=["csv"])
    if archivo_subido is not None:
        df_subido = pd.read_csv(archivo_subido)
        cols_subidas = [c for c in df_subido.columns if c.startswith('AP_')]
        if len(cols_subidas) > 0:
            huella_input = df_subido[cols_subidas].iloc[[0]].values
        else:
            st.error("El archivo CSV debe contener columnas que inicien con 'AP_'.")

# ------------------------------------------------------------------------------
# 5. PREDICCIÓN Y RESULTADOS CON VALIDACIÓN DE DIMENSIONES
# ------------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

if huella_input is not None:
    # Ajustar dimensiones de la huella al número exacto que espera el StandardScaler
    if huella_input.shape[1] > n_features_esperadas:
        huella_ajustada = huella_input[:, :n_features_esperadas]
    elif huella_input.shape[1] < n_features_esperadas:
        padding = np.full((1, n_features_esperadas - huella_input.shape[1]), -105.0)
        huella_ajustada = np.hstack([huella_input, padding])
    else:
        huella_ajustada = huella_input

    # Inferencia
    huella_scaled = scaler.transform(huella_ajustada)
    prediccion_zona = model.predict(huella_scaled)[0]
    probas = model.predict_proba(huella_scaled)[0]
    confianza_max = np.max(probas) * 100
    
    with col1:
        st.subheader("🎯 Resultado del Posicionamiento")
        st.metric(label="Zona Estimada por el Modelo", value=str(prediccion_zona))
        
        if zona_real_ejemplo:
            st.info(f"**Zona Real Esperada:** `{zona_real_ejemplo}`")
            if str(prediccion_zona) == str(zona_real_ejemplo):
                st.success("✅ ¡Coincidencia exacta con la zona real!")
            else:
                st.warning("⚠️ Estimación en zona adyacente o distinta.")
        
        st.metric(label="Confianza / Probabilidad Estimada", value=f"{confianza_max:.1f}%")
        
        if confianza_max < 40:
            st.error("🚨 **ALERTA DE CONFIANZA:** La huella difiere de los patrones de calibración.")

    with col2:
        st.subheader("📊 Top 10 Señales RSSI más Fuertes")
        df_huella = pd.DataFrame({
            "Access Point": columnas_rssi[:huella_input.shape[1]],
            "RSSI (dBm)": huella_input[0]
        })
        st.dataframe(df_huella.sort_values(by="RSSI (dBm)", ascending=False).head(10), use_container_width=True)

# ------------------------------------------------------------------------------
# 6. ADVERTENCIAS Y LIMITACIONES OPERACIONALES
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("⚠️ Limitaciones Operacionales del Modelo")
st.warning("""
* **Variabilidad de Hardware:** La exactitud depende de la sensibilidad del receptor del dispositivo cliente.
* **Atenuación Dinámica:** Flujos de personas, tabiques y estado de los APs alteran la firma electromagnética.
* **Cobertura MNAV:** Limitado exclusivamente a las 16 regiones anonimizadas del museo.
""")
