import base64
from io import BytesIO

import streamlit as st
from PIL import Image

from src.preprocess import load_and_preprocess_image
from src.predict import model_exists, load_model, predict
from src.utils import get_label, get_calories

if "cam_data" not in st.session_state:
    st.session_state.cam_data = None

st.set_page_config(
    page_title="NutriScan AI v1.0",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 30%, #f5f7fa 70%, #ffffff 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* CONTENEDORES Y ALERTAS: Forzamos color de texto oscuro para que sea legible sobre el fondo blanco */
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
    
    /* Corrección del texto invisible dentro de las alertas de Streamlit */
    div[data-testid="stAlert"] p, div.stAlert p, div[data-testid="stAlert"] *, div.stAlert * {
        color: #1f2937 !important;
        font-weight: 500;
    }

    /* Corrección para los textos secundarios (Captions) que se veían transparentes */
    .stCaption, [data-testid="stCaptionContainer"], small {
        color: #4b5563 !important;
        font-weight: 500;
        font-size: 0.88rem !important;
    }

    .stButton>button {
        border-radius: 40px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        border: none;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #fff;
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.4);
    }
    .stButton>button:active {
        transform: translateY(0);
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.7);
        color: #333;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.9);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .stRadio > div {
        gap: 1rem;
        justify-content: center;
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(8px);
        border-radius: 40px;
        padding: 0.5rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.3);
        display: inline-flex;
        margin: 0 auto;
    }
    div[data-testid="stRadio"] {
        display: flex;
        justify-content: center;
    }
    .stRadio label {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.3rem 0.8rem;
        border-radius: 30px;
        transition: all 0.2s;
    }

    .stFileUploader > div {
        border-radius: 16px;
        border: 2px dashed rgba(67, 233, 123, 0.4);
        background: rgba(255,255,255,0.5);
        backdrop-filter: blur(8px);
        padding: 0.5rem;
        transition: border-color 0.3s;
    }
    .stFileUploader > div:hover {
        border-color: rgba(67, 233, 123, 0.8);
    }

    .stAlert { text-align: center; }

    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.12);
        padding: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.18);
    }

    .mini-card {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        transition: all 0.25s ease;
        height: 100%;
    }
    .mini-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    .mini-card .emoji { font-size: 2.2rem; display: block; margin-bottom: 0.3rem; }
    .mini-card .mlabel { font-size: 0.8rem; font-weight: 700; color: #4b5563; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.2rem; }
    .mini-card .mvalue { font-size: 1.1rem; font-weight: 700; color: #111827; }
    .mini-card.green { border-top: 3px solid #22c55e; }
    .mini-card.blue { border-top: 3px solid #3b82f6; }
    .mini-card.orange { border-top: 3px solid #f97316; }

    .title-glow {
        text-align: center;
        background: linear-gradient(135deg, #2e7d32, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 500;
        color: #4b5563;
        margin-top: 0;
    }

    hr {
        margin: 1.5rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(0,0,0,0.08), transparent);
    }

    @media (max-width: 768px) {
        .title-glow { font-size: 2rem; }
        .subtitle { font-size: 0.95rem; }
        .mini-card .emoji { font-size: 1.6rem; }
        .mini-card .mvalue { font-size: 0.95rem; }
        .glass-card { padding: 1rem; }
        div[data-testid="column"] { min-width: 100% !important; }
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title-glow">🥗 NutriScan AI</div>'
    '<p class="subtitle">NutriScan AI v1.0 — Modelo Predictivo de Reconocimiento de Alimentos</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    # CORREGIDO: Se añadió unsafe_allow_html=True a la apertura del div del contenedor del sidebar
    st.markdown(
        "<div style='background: rgba(255,255,255,0.5); backdrop-filter: blur(12px); "
        "border-radius: 16px; padding: 1.2rem; border: 1px solid rgba(255,255,255,0.3);'>",
        unsafe_allow_html=True
    )
    st.markdown("### 🌿 Acerca de")
    st.caption(
        "**NutriScan AI v1.0** es un clasificador avanzado de alimentos "
        "impulsado por visión computacional."
    )
    st.divider()
    st.markdown("##### 📋 Categorías")
    for emoji, items in [
        ("🍎", ["Apple Pie", "Hamburger", "French Fries", "Hot Dog", "Sushi"]),
        ("🍩", ["Donuts", "Ice Cream", "Chicken Wings", "Pizza", "Caesar Salad"]),
    ]:
        row = f"{emoji} " + " · ".join(items)
        st.markdown(
            f"<p style='font-size: 0.85rem; line-height: 1.6; margin: 0.3rem 0; color: #1f2937;'>{row}</p>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(
        "No afirma reconocer alimentos fuera de este catálogo. "
        "Calorías aproximadas por porción estándar."
    )
    st.markdown("</div>", unsafe_allow_html=True)

model_available = model_exists()

if not model_available:
    st.warning(
        "⚠️ **Modelo no encontrado.** "
        "Ejecuta el notebook de entrenamiento para generar "
        "`model/nutriscan_model.keras` antes de usar la aplicación."
    )

st.markdown(
    "<div style='display: flex; justify-content: center; margin-bottom: 0.5rem;'>",
    unsafe_allow_html=True,
)
input_mode = st.radio("", ("Subir imagen", "Usar cámara"))
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = None
image_source = None

if input_mode == "Subir imagen":
    uploaded_file = st.file_uploader(
        "Elige una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    image_source = uploaded_file
else:
    if st.session_state.cam_data is None:
        CAMERA_HTML = """
<div id="cam-wrap" style="position:relative;width:100%;max-width:500px;margin:0 auto;border-radius:20px;overflow:hidden;background:#000;min-height:320px;box-shadow:0 8px 32px rgba(0,0,0,0.2);">
    <video id="video" autoplay playsinline style="width:100%;height:auto;display:block;transform:scaleX(-1);"></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <div style="padding:16px;text-align:center;background:rgba(0,0,0,0.7);backdrop-filter:blur(8px);">
        <button id="capture-btn" style="background:linear-gradient(135deg,#43e97b,#38f9d7);color:#fff;border:none;border-radius:40px;padding:14px 44px;font-size:17px;cursor:pointer;font-weight:700;box-shadow:0 4px 20px rgba(67,233,123,0.4);transition:transform 0.15s;letter-spacing:0.5px;">📸 Capturar foto</button>
        <p id="status" style="color:#ccc;font-size:13px;margin-top:10px;font-weight:500;">Inicializando cámara trasera...</p>
    </div>
</div>
<script>
    function sendVal(v) { if (window.Streamlit) window.Streamlit.setComponentValue(v); }
    if (window.Streamlit) window.Streamlit.setComponentReady();
    var v = document.getElementById('video'), c = document.getElementById('canvas'), b = document.getElementById('capture-btn'), s = document.getElementById('status'), ms = null;
    (async function(){
        try {
            ms = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'environment', width:{ideal:640}, height:{ideal:480} } });
            v.srcObject = ms;
            await v.play();
            s.textContent = '✅ Cámara trasera lista';
            s.style.color = '#4CAF50';
        } catch(e) {
            s.textContent = '❌ Error: ' + (e.message||'cámara no disponible');
            s.style.color = '#f44336';
            b.disabled = true; b.style.opacity = '0.5';
        }
    })();
    b.onclick = function() {
        if (!ms||!v.videoWidth) { s.textContent='⚠️ Cámara no disponible'; s.style.color='#ff9800'; return; }
        c.width = v.videoWidth; c.height = v.videoHeight;
        c.getContext('2d').drawImage(v, 0, 0);
        var img = c.toDataURL('image/jpeg',0.92).split(',')[1];
        if (ms) ms.getTracks().forEach(function(t){t.stop();});
        sendVal(img);
        s.textContent = '✅ Foto capturada';
        s.style.color = '#4CAF50';
        b.textContent = '✓ Capturada'; b.disabled = true; b.style.opacity='0.5';
    };
</script>"""
        cam_result = st.components.v1.html(CAMERA_HTML, height=420)
        if cam_result:
            st.session_state.cam_data = cam_result

    if st.session_state.cam_data:
        image_source = BytesIO(base64.b64decode(st.session_state.cam_data))

if image_source is not None:
    try:
        img_rgb, img_array = load_and_preprocess_image(image_source)

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.image(img_rgb, caption="Vista previa", use_container_width=True)
            if input_mode == "Usar cámara":
                if st.button("📸 Tomar otra foto", type="secondary", use_container_width=True):
                    st.session_state.cam_data = None
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            if model_available:
                with st.spinner("🧠 Analizando imagen..."):
                    model = load_model()
                    class_idx, confidence = predict(model, img_array)

                alimento = get_label(class_idx)
                calorias = get_calories(alimento)
                conf_pct = confidence * 100
                alimento_clean = alimento.replace("_", " ").title()
                calorias_str = f"{calorias} kcal" if calorias else "N/A"

                st.markdown(
                    "<h3 style='text-align:center;margin:0 0 1.2rem 0;font-weight:700;color:#333;'>📊 Resultado del Análisis</h3>",
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

                if conf_pct < 60:
                    st.warning(
                        f"⚠️ Confianza baja ({conf_pct:.1f}%). "
                        "El resultado puede no ser confiable. "
                        "Verifica manualmente el alimento."
                    )
                else:
                    st.success("✅ Alimento detectado con alta confianza.")

                st.caption(
                    "Nota: Solo reconoce alimentos dentro de su catálogo entrenado. "
                    "Las calorías son aproximadas por porción estándar."
                )
            else:
                st.error(
                    "No se puede realizar la predicción porque el modelo no existe. "
                    "Entrena el modelo usando el notebook proporcionado."
                )

    except Exception:
        st.error(
            "⚠️ El archivo seleccionado no se pudo procesar correctamente. "
            "Por favor, asegúrate de subir una imagen válida de un alimento "
            "(JPG, PNG o WEBP)."
        )

else:
    st.info("📸 Sube una imagen o usa la cámara para comenzar.")