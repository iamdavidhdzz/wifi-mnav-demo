import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Wi-Fi Indoor Localization - Museo MNAV",
    page_icon="📍",
    layout="wide"
)

st.title("📍 Sistema de Localización en Interiores (Wi-Fi Fingerprinting)")
st.caption("Demo interactiva de estimación de zonas en el Museo Nacional de Artes Visuales mediante k-NN y perfiles RSSI.")

# ------------------------------------------------------------------------------
# 2. CARGA DE MODELO Y DATOS (CACHEADO)
# ------------------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    model = joblib.load('modelo_knn.pkl')
    scaler = joblib.load('escalador.pkl')
    # Cargar huellas de ejemplo (o dataset guardado)
    df = pd.read_csv('ejemplos_huellas.csv')
    columnas_rssi = [c for c in df.columns if c.startswith('AP_')]
    return model, scaler, df, columnas_rssi

try:
    model, scaler, df_crudo, columnas_rssi = cargar_artefactos()
except Exception as e:
    st.error(f"Error al cargar archivos necesarios: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 3. CONTROLES Y ENTRADA DE DATOS
# ------------------------------------------------------------------------------
st.sidebar.header("🕹️ Panel de Control")

modo_entrada = st.sidebar.radio(
    "Selecciona el método de entrada para la huella Wi-Fi:",
    ["Huella de prueba preseteada (Ejemplos)", "Generar huella aleatoria", "Cargar CSV"]
)

huella_input = None
zona_real_ejemplo = None

if modo_entrada == "Huella de prueba preseteada (Ejemplos)":
    zonas_disponibles = sorted(df_crudo["location"].unique())
    zona_seleccionada = st.sidebar.selectbox("Selecciona una zona de referencia:", zonas_disponibles)
    
    muestras_zona = df_crudo[df_crudo["location"] == zona_seleccionada]
    idx_sample = st.sidebar.slider("Número de muestra de la zona:", 0, len(muestras_zona) - 1, 0)
    
    huella_input = muestras_zona.iloc[idx_sample][columnas_rssi].values.reshape(1, -1)
    zona_real_ejemplo = zona_seleccionada

elif modo_entrada == "Generar huella aleatoria":
    if st.sidebar.button("🎲 Generar nueva huella"):
        huella_input = np.random.uniform(-99, -45, size=(1, 188))
    else:
        huella_input = np.random.uniform(-99, -45, size=(1, 188))

elif modo_entrada == "Cargar CSV":
    archivo_subido = st.sidebar.file_uploader("Sube un CSV con 188 columnas (AP_001 a AP_188)", type=["csv"])
    if archivo_subido is not None:
        df_subido = pd.read_csv(archivo_subido)
        cols_subidas = [c for c in df_subido.columns if c.startswith('AP_')]
        if len(cols_subidas) == 188:
            huella_input = df_subido[cols_subidas].iloc[0].values.reshape(1, -1)
        else:
            st.error(f"El archivo debe contener 188 APs. Detectados: {len(cols_subidas)}")

# ------------------------------------------------------------------------------
# 4. PREDICCIÓN Y RESULTADOS
# ------------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

if huella_input is not None:
    huella_scaled = scaler.transform(huella_input)
    prediccion_zona = model.predict(huella_scaled)[0]
    probas = model.predict_proba(huella_scaled)[0]
    confianza_max = np.max(probas) * 100
    
    with col1:
        st.subheader("🎯 Resultado del Posicionamiento")
        st.metric(label="Zona Estimada por el Modelo", value=str(prediccion_zona))
        
        if zona_real_ejemplo:
            st.info(f"**Zona Real:** `{zona_real_ejemplo}`")
            if prediccion_zona == zona_real_ejemplo:
                st.success("✅ Coincidencia exacta con la zona real.")
            else:
                st.warning("⚠️ Estimación en zona adyacente o distinta.")
        
        st.metric(label="Confianza Estimada", value=f"{confianza_max:.1f}%")
        
        if confianza_max < 40:
            st.error("🚨 ALERTA: Confianza baja. El dispositivo puede estar fuera de la zona calibrada.")

    with col2:
        st.subheader("📊 Top 10 Señales Wi-Fi más Fuertes")
        df_huella = pd.DataFrame({
            "Access Point": columnas_rssi,
            "RSSI (dBm)": huella_input[0]
        })
        st.dataframe(df_huella.sort_values(by="RSSI (dBm)", ascending=False).head(10), use_container_width=True)

# ------------------------------------------------------------------------------
# 5. ADVERTENCIAS Y LIMITACIONES OPERACIONALES
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("⚠️ Limitaciones Operacionales del Modelo")
st.warning("""
* **Variabilidad de Hardware:** El desempeño depende de la sensibilidad de la antena Wi-Fi del dispositivo cliente.
* **Condiciones Dinámicas:** La atenuación cambia según flujo de visitantes, mobiliario o estado de los routers.
* **Alcance:** Exclusivo para las 16 zonas anonimizadas del Museo MNAV.
""")
