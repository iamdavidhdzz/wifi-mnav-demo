import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

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
        width: 100%;
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
    st.markdown('<h1 class="main-header">Localización Wi-Fi en Interiores (Fingerprinting)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Demostrador interactivo de posicionamiento en el Museo MNAV | Universidad Externado de Colombia</p>', unsafe_allow_html=True)

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
    st.error(f"🚨 Error cargando los archivos del modelo: {e}")
    st.stop()

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
# 5. PREDICCIÓN Y RESULTADOS INTERACTIVOS
# ------------------------------------------------------------------------------
if huella_input is not None:
    # Ajuste de dimensión para el scaler
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

    col_izq, col_der = st.columns([1, 1])

    # --------------------------------------------------------------------------
    # COLUMNA IZQUIERDA: PLANO INTERACTIVO Y PREDICCIÓN
    # --------------------------------------------------------------------------
    with col_izq:
        st.subheader("📍 Posicionamiento en el Plano del Museo")
        st.metric(label="Zona Estimada por el Modelo", value=str(prediccion_zona))
        
        if zona_real_ejemplo:
            if str(prediccion_zona) == str(zona_real_ejemplo):
                st.success(f"✅ ¡Coincidencia exacta con la zona real: `{zona_real_ejemplo}`!")
            else:
                st.warning(f"⚠️ Estimación desviada. Real: `{zona_real_ejemplo}` | Predicha: `{prediccion_zona}`")

        # Mapeo de coordenadas 2D para las 16 regiones
        coords = {
            "location_1": (1, 4),  "location_2": (2, 4),  "location_3": (3, 4),  "location_4": (4, 4),
            "location_5": (1, 3),  "location_6": (2, 3),  "location_7": (3, 3),  "location_8": (4, 3),
            "location_9": (1, 2),  "location_10": (2, 2), "location_11": (3, 2), "location_12": (4, 2),
            "location_13": (1, 1), "location_14": (2, 1), "location_15": (3, 1), "location_16": (4, 1),
        }

        data_mapa = []
        for k, v in coords.items():
            es_predicha = (k == str(prediccion_zona))
            data_mapa.append({
                "Zona": k, "X": v[0], "Y": v[1],
                "Estado": "Zona Detectada" if es_predicha else "Otras Salas",
                "Tamaño": 30 if es_predicha else 14
            })
        df_mapa = pd.DataFrame(data_mapa)

        fig_mapa = px.scatter(
            df_mapa, x="X", y="Y", text="Zona", color="Estado", size="Tamaño",
            color_discrete_map={"Zona Detectada": "#005A36", "Otras Salas": "#CBD5E0"},
            title="🗺️ Distribución Espacial de Salas MNAV"
        )
        fig_mapa.update_traces(textposition='top center', font=dict(size=12, color="black"))
        fig_mapa.update_layout(
            height=380,
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_mapa, use_container_width=True)

    # --------------------------------------------------------------------------
    # COLUMNA DERECHA: TACÓMETRO DE CONFIANZA Y BARRAS RSSI
    # --------------------------------------------------------------------------
    with col_der:
        st.subheader("📊 Diagnóstico y Certidumbre")
        
        # Tacómetro Gauge de Confianza
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confianza_max,
            number = {'suffix': "%"},
            title = {'text': "Nivel de Certeza ($k$-NN)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#005A36"},
                'steps': [
                    {'range': [0, 40], 'color': "#FEB2B2"},
                    {'range': [40, 75], 'color': "#FEEBC8"},
                    {'range': [75, 100], 'color': "#C6F6D5"}
                ],
            }
        ))
        fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if confianza_max < 40:
            st.error("🚨 **ALERTA DE SEGURIDAD:** Confianza baja. Dispositivo probablemente fuera de cobertura o en zona sin calibrar.")

        # Top 10 RSSI
        st.markdown("##### 📶 Top 10 Access Points más Fuertes")
        df_huella = pd.DataFrame({
            "Access Point": columnas_rssi[:huella_input.shape[1]],
            "RSSI (dBm)": huella_input[0]
        }).sort_values(by="RSSI (dBm)", ascending=False).head(10)

        fig_rssi = px.bar(
            df_huella, x="RSSI (dBm)", y="Access Point", orientation='h',
            color="RSSI (dBm)", color_continuous_scale="Greens"
        )
        fig_rssi.update_layout(
            height=260,
            yaxis={'categoryorder':'total ascending'},
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_rssi, use_container_width=True)

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
