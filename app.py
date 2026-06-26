import streamlit as st
from PIL import Image
from io import BytesIO

from src.preprocess import load_and_preprocess_image
from src.predict import model_exists, load_model, predict
from src.utils import get_label, get_calories

# Configuración de la página
st.set_page_config(
    page_title="NutriScan AI v1.0",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# 🎨 DISEÑO E INYECCIÓN DE CSS PARA CONTROL TOTAL DE COLORES (ANTI-MODO OSCURO)
# =========================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Fondo general de la aplicación */
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 30%, #f5f7fa 70%, #ffffff 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* AREA PRINCIPAL (TEXTOS OSCUROS OBLIGATORIOS) */
    [data-testid="stMain"] p, 
    [data-testid="stMain"] label, 
    [data-testid="stMain"] span:not(.emoji), 
    [data-testid="stMain"] small,
    [data-testid="stHeader"] * {
        color: #1f2937 !important;
    }

    /* BARRA LATERAL (FONDO OSCURO Y TEXTOS CLAROS OBLIGATORIOS) */
    [data-testid="stSidebar"] {
        background-color: #1f2937 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] small {
        color: #f3f4f6 !important;
    }

    [data-testid="stSidebar"] .stCaption, 
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #9ca3af !important;
    }

    /* TARJETAS, CONTENEDORES Y ALERTAS */
    div[data-testid="stImage"], div[data-testid="stAlert"], div.stAlert,
    div.stInfo, div.stWarning, div.stError, div.stSuccess {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.12);
        padding: 1.2rem;
    }
    
    div[data-testid="stAlert"] *, div.stAlert *, .stAlert p {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }

    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .stCaption, .stCaption * {
        color: #4b5563 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.12);
        padding: 1.5rem;
    }

    /* Tarjetas de métricas */
    .mini-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    .mini-card .emoji { font-size: 2.2rem; display: block; margin-bottom: 0.3rem; }
    .mini-card .mlabel { font-size: 0.8rem; font-weight: 700; color: #4b5563 !important; text-transform: uppercase; }
    .mini-card .mvalue { font-size: 1.1rem; font-weight: 700; color: #111827 !important; }
    .mini-card.green { border-top: 3px solid #22c55e; }
    .mini-card.blue { border-top: 3px solid #3b82f6; }
    .mini-card.orange { border-top: 3px solid #f97316; }

    /* BOTONES Y COMPONENTES NATIVOS */
    .stButton>button {
        border-radius: 40px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        border: none;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
    }
    .stButton>button * { color: #ffffff !important; }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.4);
    }

    .stRadio > div {
        gap: 1rem;
        justify-content: center;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        border-radius: 40px;
        padding: 0.5rem 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
        display: inline-flex;
    }
    .stRadio label, .stRadio label span {
        font-weight: 600 !important;
        color: #1f2937 !important;
    }

    .stFileUploader > div, .stCameraInput > div {
        border-radius: 16px;
        border: 2px dashed rgba(67, 233, 123, 0.4);
        background: rgba(255, 255, 255, 0.6);
        padding: 0.5rem;
    }

    .stCameraInput span, .stCameraInput button {
        color: #1f2937 !important;
    }

    .title-glow {
        text-align: center;
        background: linear-gradient(135deg, #2e7d32, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 500;
        color: #4b5563 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================================
# HEADER PRINCIPAL
# =========================================================================
st.markdown(
    '<div class="title-glow">🥗 NutriScan AI</div>'
    '<p class="subtitle">NutriScan AI v1.0 — Modelo Predictivo de Reconocimiento de Alimentos</p>',
    unsafe_allow_html=True,
)

# =========================================================================
# BARRA LATERAL (SIDEBAR)
# =========================================================================
with st.sidebar:
    st.markdown("<h3 style='margin-top: 0;'>🌿 Acerca de</h3>", unsafe_allow_html=True)
    st.caption(
        "**NutriScan AI v1.0** es un clasificador avanzado de alimentos "
        "impulsado por visión computacional."
    )
    st.divider()
    st.markdown("<h5>📋 Categorías</h5>", unsafe_allow_html=True)
    
    categorias = [
        ("🍎", ["Apple Pie", "Hamburger", "French Fries", "Hot Dog", "Sushi"]),
        ("🍩", ["Donuts", "Ice Cream", "Chicken Wings", "Pizza", "Caesar Salad"]),
    ]
    for emoji, items in categorias:
        row = f"{emoji} " + " · ".join(items)
        st.markdown(
            f"<p style='font-size: 0.85rem; line-height: 1.6; margin: 0.3rem 0;'>{row}</p>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(
        "No afirma reconocer alimentos fuera de este catálogo. "
        "Calorías aproximadas por porción estándar."
    )

# =========================================================================
# COMPROBACIÓN DE MODELO
# =========================================================================
model_available = model_exists()

if not model_available:
    st.warning(
        "⚠️ **Modelo no encontrado.** "
        "Ejecuta el notebook de entrenamiento para generar "
        "`model/nutriscan_model.keras` antes de usar la aplicación."
    )

# =========================================================================
# SELECTOR DE MODO DE ENTRADA Y CAPTURA
# =========================================================================
st.markdown(
    "<div style='display: flex; justify-content: center; margin-bottom: 0.5rem;'>",
    unsafe_allow_html=True,
)
input_mode = st.radio("", ("Subir imagen", "Usar cámara"))
st.markdown("</div>", unsafe_allow_html=True)

image_source = None

if input_mode == "Subir imagen":
    uploaded_file = st.file_uploader(
        "Elige una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded_file is not None:
        image_source = uploaded_file
else:
    cam_file = st.camera_input("Toma una foto de tu platillo")
    if cam_file is not None:
        image_source = cam_file

# =========================================================================
# LÓGICA DE DETECCIÓN Y PROCESAMIENTO
# =========================================================================
if image_source is not None:
    try:
        img_rgb, img_array = load_and_preprocess_image(image_source)

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.image(img_rgb, caption="Imagen cargada con éxito", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            if model_available:
                with st.spinner("🧠 Analizando imagen..."):
                    model = load_model()
                    class_idx, confidence = predict(model, img_array)

                conf_pct = confidence * 100

                # -----------------------------------------------------------------
                # FILTRO DE VALIDACIÓN: ¿Es un objeto o foto sin relación con comida?
                # -----------------------------------------------------------------
                if conf_pct < 50.0:
                    st.markdown(
                        "<h3 style='text-align: center; margin-top: 2rem; font-weight: 700; color: #dc2626;'>🔍 Alimento No Detectado</h3>",
                        unsafe_allow_html=True,
                    )
                    st.error(
                        "❌ **No se encontró ningún alimento en la imagen.**\n\n"
                        "Por favor, asegúrate de subir o tomar una foto nítida de una comida "
                        "que pertenezca a nuestro catálogo (por ejemplo: Pizza, Hamburguesa, Sushi, etc.)."
                    )
                    st.info(
                        "💡 *Consejo: Evita tomarle fotos a personas, objetos del entorno o capturas "
                        "con mala iluminación.*"
                    )
                else:
                    # Si pasa la validación, muestra los resultados normalmente
                    alimento = get_label(class_idx)
                    calorias = get_calories(alimento)
                    alimento_clean = alimento.replace("_", " ").title()
                    calorias_str = f"{calorias} kcal" if calorias else "N/A"

                    st.markdown(
                        "<h3 style='text-align: center; margin: 0 0 1.2rem 0; font-weight: 700; color: #1f2937;'>📊 Resultado del Análisis</h3>",
                        unsafe_allow_html=True,
                    )

                    mc1, mc2, mc3 = st.columns(3, gap="small")
                    with mc1:
                        st.markdown(
                            f"<div class='mini-card green'>"
                            f"<span class='emoji'>🥘</span>"
                            f"<div class='mlabel'>Alimento</div>"
                            f"<div class='mvalue'>{alimento_clean}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with mc2:
                        st.markdown(
                            f"<div class='mini-card blue'>"
                            f"<span class='emoji'>🎯</span>"
                            f"<div class='mlabel'>Confianza</div>"
                            f"<div class='mvalue'>{conf_pct:.1f}%</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with mc3:
                        st.markdown(
                            f"<div class='mini-card orange'>"
                            f"<span class='emoji'>🔥</span>"
                            f"<div class='mlabel'>Calorías</div>"
                            f"<div class='mvalue'>{calorias_str}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    if conf_pct < 65:
                        st.warning(
                            f"⚠️ Confianza moderada ({conf_pct:.1f}%). "
                            "El resultado podría variar ligeramente."
                        )
                    else:
                        st.success("✅ Alimento detectado con alta confianza.")

                st.caption(
                    "Nota: Las calorías son aproximadas por porción estándar basadas en el catálogo entrenado."
                )
            else:
                st.error(
                    "No se puede realizar la predicción porque el modelo no existe. "
                    "Entrena el modelo usando el notebook proporcionado."
                )

    except Exception as e:
        st.error(
            "⚠️ El archivo seleccionado no se pudo procesar correctamente. "
            "Por favor, asegúrate de que la imagen sea un formato válido."
        )

else:
    st.info("📸 Sube una imagen o activa la cámara para comenzar.")