import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN E IDENTIDAD VISUAL
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Localización Wi-Fi en interiores | Externado",
    page_icon="📍",
    layout="wide",
)

VERDE = "#005A36"
VERDE_SUAVE = "#9DB8AB"
PAPEL = "#F4F7F5"
TINTA = "#1F2933"
OCRE = "#B7791F"
ROJO = "#B3261E"
SIN_SENAL = -100.0  # valor con el que se entrenó: AP no detectado

# Compatibilidad entre versiones de Streamlit (use_container_width está en retiro)
def _version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)

STRETCH = {"width": "stretch"} if _version() >= (1, 50) else {"use_container_width": True}


AVISOS = {
    "ok": ("#E3F2E9", VERDE, "#00402A"),
    "error": ("#FBE6E3", ROJO, "#7A1913"),
    "warn": ("#FCF0D4", OCRE, "#5E3F0D"),
    "info": ("#E7EEF3", "#52606D", TINTA),
}


def aviso(tipo, texto):
    """Mensaje con colores propios: se lee igual en tema claro u oscuro."""
    fondo, borde, color = AVISOS[tipo]
    st.markdown(
        f"<div style='background:{fondo};border-left:5px solid {borde};padding:.7rem .95rem;"
        f"border-radius:4px;margin:.4rem 0;color:{color} !important'>{texto}</div>",
        unsafe_allow_html=True,
    )


def tabla_html(df, alto=300):
    html = df.to_html(index=False, border=0, classes="tabla", float_format=lambda v: f"{v:.0f}")
    st.markdown(
        f"<div class='tabla-wrap' style='max-height:{alto}px'>{html.replace(chr(10), '')}</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Source+Sans+3:wght@400;600&display=swap');
    :root {{ color-scheme: light; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{ background-color: #FFFFFF !important; }}
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{ background-color: {PAPEL} !important; }}
    section[data-testid="stSidebar"] {{ border-right: 1px solid #D9E2DC; }}
    .stApp {{ font-family: 'Source Sans 3', sans-serif; }}
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"] {{ color: {TINTA} !important; }}
    h1, h2, h3 {{ font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; }}
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    .stApp p.titulo {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.1rem;
                       line-height: 1.15; color: {VERDE} !important; margin: 0; }}
    .stApp p.bajada {{ color: #3E4C59 !important; margin: .35rem 0 0 0; font-size: 1.05rem; }}
    .stApp p.zona-grande {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 3.4rem; font-weight: 700;
                            line-height: 1; color: {VERDE} !important; margin: .1rem 0 0 0; }}
    .stApp p.zona-grande.baja {{ color: {ROJO} !important; }}
    .stApp p.rotulo {{ font-size: .95rem; color: #3E4C59 !important; margin: 0; }}
    .stApp [data-testid="stMetricValue"] {{ font-family: 'Bricolage Grotesque', sans-serif; }}
    .stApp [data-testid="stSliderThumbValue"] {{ color: {VERDE} !important; }}
    .stApp [data-baseweb="slider"] [role="slider"] {{ background-color: {VERDE} !important; }}
    .stApp button[data-baseweb="tab"][aria-selected="true"] p {{ color: {VERDE} !important; font-weight: 600; }}
    .stApp [data-baseweb="tab-highlight"] {{ background-color: {VERDE} !important; }}
    .stApp .stButton > button, .stApp .stDownloadButton > button {{
        background-color: {VERDE} !important; border: 0; border-radius: 6px; width: 100%; }}
    .stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {{ background-color: #00422A !important; }}
    .stApp .stButton > button p, .stApp .stDownloadButton > button p {{ color: #FFFFFF !important; font-weight: 600; }}
    .stApp .stButton > button:focus-visible, .stApp .stDownloadButton > button:focus-visible {{
        outline: 3px solid {OCRE}; outline-offset: 2px; }}
    .stApp [data-baseweb="select"] > div {{ background-color: #FFFFFF !important; border: 1px solid #BFCBC5 !important; }}
    .stApp [data-baseweb="select"] * {{ color: {TINTA} !important; }}
    .stApp [data-testid="stFileUploaderDropzone"] {{ background-color: #FFFFFF !important; border: 1px dashed {VERDE_SUAVE}; }}
    .stApp [data-testid="stFileUploaderDropzone"] button {{ background-color: #FFFFFF !important; border: 1px solid #BFCBC5 !important; }}
    .tabla-wrap {{ overflow: auto; border: 1px solid #D9E2DC; border-radius: 6px; }}
    .tabla {{ width: 100%; border-collapse: collapse; font-size: .95rem; }}
    .tabla th {{ position: sticky; top: 0; background: {PAPEL}; color: {TINTA}; text-align: left; padding: .4rem .7rem; }}
    .tabla td {{ color: {TINTA}; padding: .35rem .7rem; border-top: 1px solid #E6ECE8; }}
        </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. ENCABEZADO
# ------------------------------------------------------------------------------
col_logo, col_titulo = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    if Path("image_4b1621.png").exists():
        st.image("image_4b1621.png", width=150)
    else:
        st.markdown(f"<p class='titulo' style='font-size:1.4rem'>EXTERNADO</p>", unsafe_allow_html=True)
with col_titulo:
    st.markdown(
        "<p class='titulo'>¿En qué sala del museo está el dispositivo?</p>"
        "<p class='bajada'>Estimación a partir de las señales Wi-Fi que recibe, "
        "con un modelo k-NN entrenado en el Museo Nacional de Artes Visuales.</p>",
        unsafe_allow_html=True,
    )
st.write("")

# ------------------------------------------------------------------------------
# 3. CARGA DE MODELO Y DATOS (una sola vez por sesión del servidor)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo():
    return joblib.load("modelo_knn.pkl"), joblib.load("escalador.pkl")


@st.cache_data(show_spinner=False)
def cargar_huellas():
    return pd.read_csv("ejemplos_huellas.csv")


try:
    model, scaler = cargar_modelo()
    df_ref = cargar_huellas()
except FileNotFoundError as e:
    aviso("error", 
        f"No se encontró el archivo <code>{e.filename}</code>. Deja <code>modelo_knn.pkl</code>, <code>escalador.pkl</code> "
        "y <code>ejemplos_huellas.csv</code> en la misma carpeta que <code>app.py</code>."
    )
    st.stop()
except Exception as e:
    aviso("error", f"No se pudo cargar el modelo: {e}")
    st.stop()

AP_COLS = [c for c in df_ref.columns if c.startswith("AP_")]
TIENE_NOMBRES = hasattr(scaler, "feature_names_in_")
if TIENE_NOMBRES:
    FEATURES = list(scaler.feature_names_in_)
else:
    FEATURES = AP_COLS[: getattr(scaler, "n_features_in_", len(AP_COLS))]
CLASES = np.array([str(c) for c in model.classes_])


def etq(zona: str) -> str:
    return re.sub(r"^location_", "Zona ", str(zona))


def a_matriz(df_in: pd.DataFrame) -> pd.DataFrame:
    """Alinea por NOMBRE de AP con lo que vio el modelo; lo que falta o es 0 = sin señal."""
    X = df_in.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(SIN_SENAL)
    return X.mask(X == 0, SIN_SENAL).astype(float)


def predecir(X: pd.DataFrame) -> np.ndarray:
    Xs = scaler.transform(X if TIENE_NOMBRES else X.to_numpy())
    return model.predict_proba(Xs)


# ------------------------------------------------------------------------------
# 4. BARRA LATERAL
# ------------------------------------------------------------------------------
st.sidebar.markdown("### Origen de la huella")
modo = st.sidebar.radio(
    "Origen",
    ["Muestra de referencia", "Simulación con ruido", "Subir CSV"],
    label_visibility="collapsed",
)

zona_real = None
lote, lote_real = None, None

if modo == "Muestra de referencia":
    zonas = sorted(df_ref["location"].astype(str).unique(), key=lambda z: int(re.sub(r"\D", "", z) or 0))
    zona_sel = st.sidebar.selectbox("Zona real", zonas, format_func=etq)
    muestras = df_ref[df_ref["location"].astype(str) == zona_sel]
    i = 0
    if len(muestras) > 1:
        i = st.sidebar.slider("Muestra de la zona", 0, len(muestras) - 1, 0)
    huella = a_matriz(muestras.iloc[[i]])
    zona_real = zona_sel

elif modo == "Simulación con ruido":
    st.sidebar.caption(
        "Toma una huella real al azar y le agrega ruido y pérdida de señal, "
        "como pasaría con otro celular o con gente en las salas."
    )
    sigma = st.sidebar.slider("Ruido (dB)", 0.0, 12.0, 4.0, 0.5)
    perdida = st.sidebar.slider("APs que se pierden", 0, 60, 10, format="%d %%") / 100
    if "semilla" not in st.session_state:
        st.session_state.semilla = 0
    if st.sidebar.button("Nueva simulación"):
        st.session_state.semilla += 1

    rng = np.random.default_rng(st.session_state.semilla)
    fila = df_ref.sample(1, random_state=int(rng.integers(1_000_000)))
    zona_real = str(fila["location"].iloc[0])
    base = a_matriz(fila).to_numpy()
    detectado = base > SIN_SENAL
    ruidoso = np.clip(base + rng.normal(0, sigma, base.shape), -99, -30)
    base = np.where(detectado, ruidoso, SIN_SENAL)
    base = np.where(detectado & (rng.random(base.shape) < perdida), SIN_SENAL, base)
    huella = pd.DataFrame(base, columns=FEATURES)

else:  # Subir CSV
    archivo = st.sidebar.file_uploader("CSV con columnas AP_…", type=["csv"])
    if archivo is None:
        aviso("info", "Sube un CSV con columnas <code>AP_…</code> en la barra lateral para ubicar una huella.")
        st.stop()
    try:
        df_sub = pd.read_csv(archivo)
    except Exception as e:
        aviso("error", f"No se pudo leer el archivo: {e}")
        st.stop()
    presentes = [c for c in FEATURES if c in df_sub.columns]
    if not presentes:
        aviso("error", "El CSV no tiene ninguna de las columnas <code>AP_…</code> que usa el modelo.")
        st.stop()
    st.sidebar.caption(
        f"{len(presentes)} de {len(FEATURES)} APs encontrados. "
        f"Los demás se toman como {SIN_SENAL:.0f} dBm."
    )
    lote = a_matriz(df_sub)
    i = 0
    if len(lote) > 1:
        i = st.sidebar.slider("Fila a localizar", 0, len(lote) - 1, 0)
    huella = lote.iloc[[i]]
    if "location" in df_sub.columns:
        lote_real = df_sub["location"].astype(str)
        zona_real = lote_real.iloc[i]

st.sidebar.markdown("---")
umbral = st.sidebar.slider("Alertar si la certeza es menor a", 10, 90, 40, format="%d %%")

# ------------------------------------------------------------------------------
# 5. PREDICCIÓN
# ------------------------------------------------------------------------------
proba = predecir(huella)[0]
orden = np.argsort(proba)[::-1]
pred = CLASES[orden[0]]
certeza = proba[orden[0]] * 100
n_detectados = int((huella.iloc[0] > SIN_SENAL).sum())
baja = certeza < umbral


def plano(proba, pred, real):
    """Plano de salas coloreado por la fracción de vecinos que votó por cada una."""
    n = len(CLASES)
    ncols = 4 if n == 16 else int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    def clave(i):
        m = re.search(r"(\d+)$", CLASES[i])
        return int(m.group(1)) if m else i

    z = np.full((nrows, ncols), np.nan)
    pos = {}
    for rank, i in enumerate(sorted(range(n), key=clave)):
        r, c = divmod(rank, ncols)
        z[r, c] = proba[i]
        pos[CLASES[i]] = (r, c)

    fig = go.Figure(
        go.Heatmap(
            z=z, zmin=0, zmax=1, xgap=6, ygap=6, showscale=False,
            colorscale=[[0, "#EAEFEC"], [1, VERDE]],
            hoverinfo="skip",
        )
    )
    for i, zona in enumerate(CLASES):
        r, c = pos[zona]
        fuerte = proba[i] > 0.55
        pct = f"{proba[i] * 100:.0f} %" if proba[i] > 0 else ""
        fig.add_annotation(
            x=c, y=r, showarrow=False,
            text=f"<b>{etq(zona)}</b><br>{pct}",
            font=dict(size=13, color="white" if fuerte else TINTA),
        )
    for zona, color, ancho in ((real, OCRE, 5), (pred, TINTA, 2)):
        if zona in pos:
            r, c = pos[zona]
            fig.add_shape(
                type="rect", x0=c - .5, x1=c + .5, y0=r - .5, y1=r + .5,
                line=dict(color=color, width=ancho), fillcolor="rgba(0,0,0,0)",
            )
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed", scaleanchor="x"),
    )
    return fig


def _posiciones():
    n = len(CLASES)
    ncols = 4 if n == 16 else int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    def clave(i):
        m = re.search(r"(\d+)$", CLASES[i])
        return int(m.group(1)) if m else i

    pos = {CLASES[i]: divmod(rank, ncols) for rank, i in enumerate(sorted(range(n), key=clave))}
    return pos, nrows, ncols


def _mezcla(p):
    """Color entre gris verdoso claro (p=0) y el verde institucional (p=1)."""
    a, b = np.array([0xDD, 0xE5, 0xE0]), np.array([0x00, 0x5A, 0x36])
    r, g, bl = (a + (b - a) * p).round().astype(int)
    return f"rgb({r},{g},{bl})"


def plano_3d(proba, pred, real):
    """Salas como prismas: altura y color = fracción de vecinos que votó por cada una."""
    pos, nrows, ncols = _posiciones()
    I = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    J = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    K = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    fig = go.Figure()
    alturas = {}
    for i, zona in enumerate(CLASES):
        r, c = pos[zona]
        x, y = c, nrows - 1 - r
        h = 0.06 + 1.2 * proba[i]
        alturas[zona] = h
        x0, x1, y0, y1 = x - .44, x + .44, y - .44, y + .44
        fig.add_trace(
            go.Mesh3d(
                x=[x0, x1, x1, x0, x0, x1, x1, x0], y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[0, 0, 0, 0, h, h, h, h], i=I, j=J, k=K,
                color=_mezcla(proba[i]), flatshading=True,
                lighting=dict(ambient=.65, diffuse=.8, specular=.1),
                hoverinfo="text", hovertext=f"{etq(zona)}: {proba[i] * 100:.0f} % de los vecinos",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[pos[z][1] for z in CLASES], y=[nrows - 1 - pos[z][0] for z in CLASES],
            z=[alturas[z] + .12 for z in CLASES], mode="text",
            text=[etq(z) + (f" ({proba[i] * 100:.0f} %)" if proba[i] > 0 else "") for i, z in enumerate(CLASES)],
            textfont=dict(size=12, color=TINTA), hoverinfo="skip", showlegend=False,
        )
    )
    for zona, color, ancho in ((real, OCRE, 9), (pred, TINTA, 5)):
        if zona in pos:
            r, c = pos[zona]
            x, y, h = c, nrows - 1 - r, alturas[zona] + .01
            fig.add_trace(
                go.Scatter3d(
                    x=[x - .44, x + .44, x + .44, x - .44, x - .44],
                    y=[y - .44, y - .44, y + .44, y + .44, y - .44], z=[h] * 5,
                    mode="lines", line=dict(color=color, width=ancho),
                    hoverinfo="skip", showlegend=False,
                )
            )
    oculto = dict(visible=False)
    fig.update_layout(
        height=470, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=oculto, yaxis=oculto, zaxis=dict(visible=False, range=[0, 2]),
            aspectmode="manual", aspectratio=dict(x=ncols / 2.2, y=nrows / 2.2, z=.55),
            camera=dict(eye=dict(x=0.1, y=-1.9, z=1.25)), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def barras_zonas(proba, orden):
    top = orden[:3]
    fig = go.Figure(
        go.Bar(
            x=proba[top] * 100, y=[etq(CLASES[i]) for i in top], orientation="h",
            marker_color=[VERDE] + [VERDE_SUAVE] * (len(top) - 1),
            text=[f"{proba[i] * 100:.0f} %" for i in top], textposition="outside", cliponaxis=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=170, margin=dict(l=0, r=40, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False, range=[0, 110]),
        yaxis=dict(autorange="reversed"), font=dict(size=13, color=TINTA),
    )
    return fig


def barras_aps(huella):
    s = huella.iloc[0]
    s = s[s > SIN_SENAL].sort_values(ascending=False).head(10)
    fig = go.Figure(
        go.Bar(
            x=s.values, y=s.index, orientation="h", marker_color=VERDE,
            hovertemplate="%{y}: %{x:.0f} dBm<extra></extra>",
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=0, r=10, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"),
        xaxis=dict(title="RSSI (dBm), más cerca de 0 = más fuerte", range=[-100, -30]),
        font=dict(size=12, color=TINTA),
    )
    return fig


# ------------------------------------------------------------------------------
# 6. RESULTADOS
# ------------------------------------------------------------------------------
tab_loc, tab_huella, tab_modelo = st.tabs(["Localización", "Huella RSSI", "Modelo y límites"])

with tab_loc:
    col_plano, col_info = st.columns([3, 2], gap="large")

    with col_plano:
        vista = st.radio("Vista del plano", ["3D", "2D"], horizontal=True, label_visibility="collapsed")
        if vista == "3D":
            st.plotly_chart(plano_3d(proba, pred, zona_real), config={"displayModeBar": False}, **STRETCH)
            leyenda = "Arrastra para girar y usa la rueda para acercar. Altura y color indican la fracción de vecinos del k-NN que eligió cada sala."
        else:
            st.plotly_chart(plano(proba, pred, zona_real), config={"displayModeBar": False}, **STRETCH)
            leyenda = "Cada sala se colorea según la fracción de vecinos del k-NN que la eligió."
        if zona_real:
            leyenda += " Contorno ocre: zona real. Contorno oscuro: zona estimada."
        st.caption(leyenda)

    with col_info:
        st.markdown("<p class='rotulo'>Sala estimada</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='zona-grande{' baja' if baja else ''}'>{etq(pred)}</p>",
            unsafe_allow_html=True,
        )
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Certeza", f"{certeza:.0f} %")
        m2.metric("APs detectados", f"{n_detectados} de {len(FEATURES)}")

        if n_detectados == 0:
            aviso("error", "El dispositivo no detecta ningún AP: está fuera de cobertura.")
        elif baja:
            aviso("error", 
                "Certeza baja. El dispositivo puede estar fuera de cobertura "
                "o en una zona que el modelo no calibró."
            )
        if zona_real:
            if pred == zona_real:
                aviso("ok", f"Acierto: la zona real es {etq(zona_real)}.")
            else:
                aviso("warn", f"Error: la zona real es {etq(zona_real)} y el modelo estimó {etq(pred)}.")

        st.markdown("**Zonas más probables**")
        st.plotly_chart(barras_zonas(proba, orden), config={"displayModeBar": False}, **STRETCH)

with tab_huella:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("**Los 10 APs con señal más fuerte**")
        if n_detectados:
            st.plotly_chart(barras_aps(huella), config={"displayModeBar": False}, **STRETCH)
        else:
            aviso("info", "No hay APs detectados en esta huella.")
    with c2:
        st.markdown("**Todos los APs detectados**")
        det = huella.iloc[0]
        det = det[det > SIN_SENAL].sort_values(ascending=False)
        tabla_html(det.rename("RSSI (dBm)").rename_axis("Access Point").reset_index(), 300)

    if lote is not None and len(lote) > 1:
        st.markdown("---")
        st.markdown("**Localizar todo el archivo**")
        if st.checkbox("Calcular las filas del CSV"):
            P = predecir(lote)
            res = pd.DataFrame(
                {
                    "fila": np.arange(len(lote)),
                    "zona_estimada": CLASES[P.argmax(1)],
                    "certeza_%": (P.max(1) * 100).round(0).astype(int),
                }
            )
            if lote_real is not None:
                res["zona_real"] = lote_real.values
                st.metric("Exactitud en el archivo", f"{(res.zona_real == res.zona_estimada).mean() * 100:.1f} %")
            tabla_html(res, 300)
            st.download_button(
                "Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8"),
                file_name="localizacion_resultados.csv", mime="text/csv",
            )

with tab_modelo:
    k = getattr(model, "n_neighbors", None)
    cA, cB, cC = st.columns(3)
    cA.metric("Vecinos (k)", k if k is not None else "n/d")
    cB.metric("APs de entrada", len(FEATURES))
    cC.metric("Zonas", len(CLASES))
    st.markdown(
        f"""
**Cómo se calcula la certeza.** El modelo busca las {k if k else 'k'} huellas de entrenamiento más
parecidas y cuenta cuántas pertenecen a cada zona. Esa fracción es la certeza: no es una probabilidad
calibrada, por eso toma valores discretos (con k = 5, de 20 en 20 %).

**Cómo se prepara la huella.** Los APs que el dispositivo no ve (o que llegan con 0) se toman como
{SIN_SENAL:.0f} dBm, igual que en el entrenamiento. Las columnas se alinean por nombre, así que un CSV
con el orden distinto o con APs faltantes no desplaza los datos.
"""
    )
    aviso(
        "info",
        "<b>Límites del modelo</b><ul style='margin:.3rem 0 0 1.1rem;padding:0'>"
        "<li>La exactitud depende de la sensibilidad del receptor del dispositivo.</li>"
        "<li>Personas, tabiques y el estado de los APs cambian la señal.</li>"
        "<li>Solo cubre las 16 regiones anonimizadas del museo.</li></ul>",
    )
