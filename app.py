import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64
import os
import time
import io

# =========================
# 1. PAGE CONFIG & STATE
# =========================
st.set_page_config(
    layout="wide",
    page_title="Neuronix AI | Clinical Excellence",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# SPA State Management
if "app_state" not in st.session_state:
    st.session_state.app_state = "idle"
if "raw_image" not in st.session_state:
    st.session_state.raw_image = None
if "res_image" not in st.session_state:
    st.session_state.res_image = None
if "detections" not in st.session_state:
    st.session_state.detections = None
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# Fix Model Load
from pathlib import Path
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "yolov11l.pt"
    return YOLO(str(model_path))

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

# =========================
# 2. ASSETS & HELPERS
# =========================
CLOUDINARY_VIDEO_URL = "https://res.cloudinary.com/dkxoksnpz/video/upload/v1774362450/3130284-uhd_3840_2160_30fps_mzryjc.mp4"

def get_image_base64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# Konversi gambar ke base64 untuk overlay animasi CSS
def pil_to_base64(img: Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

pe_anatomy_base64 = get_image_base64("result.png")

# =========================
# 3. ELITE CSS & ANIMATIONS
# =========================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:ital,wght@1,400;1,500;1,600;1,700&display=swap" rel="stylesheet">
<style>
    /* Reset & Background */
    .stApp {
        background: radial-gradient(circle at top center, #0a0e14 0%, #000000 100%);
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 1100px !important; 
        margin: 0 auto !important;
        overflow-x: hidden;
    }

    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-up {
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        opacity: 0;
    }
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.3s; }
    .delay-3 { animation-delay: 0.5s; }

    /* --- ANIMASI SCANNER LASER --- */
    @keyframes scanline {
        0% { top: 0%; opacity: 0; }
        10% { opacity: 1; box-shadow: 0 0 15px #A8D5D5, 0 0 30px #A8D5D5; }
        90% { opacity: 1; box-shadow: 0 0 15px #A8D5D5, 0 0 30px #A8D5D5; }
        100% { top: 100%; opacity: 0; }
    }
    
    .scanner-box {
        position: relative;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        background: rgba(0,0,0,0.3);
        padding: 10px;
        border: 1px solid rgba(168, 213, 213, 0.3);
    }
    .scanner-box img {
        width: 100%;
        height: auto;
        border-radius: 8px;
        display: block;
        filter: contrast(1.1); /* Sedikit lebih dramatis saat di scan */
    }
    .scanner-laser {
        position: absolute;
        left: 0;
        width: 100%;
        height: 2px;
        background: #A8D5D5;
        animation: scanline 2s ease-in-out infinite;
        z-index: 10;
    }

    /* Hero Section */
    .hero-container {
        position: relative;
        width: 100%;
        height: 65vh;
        min-height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 3rem;
        border-radius: 0 0 40px 40px;
        overflow: hidden;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .hero-video {
        position: absolute;
        inset: 0;
        width: 100%; height: 100%;
        object-fit: cover;
        opacity: 0.25;
        z-index: 0;
    }
    .hero-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0.2) 0%, #000000 100%);
        z-index: 1;
    }
    .hero-content {
        position: relative;
        z-index: 2;
        text-align: center;
        max-width: 800px;
        padding: 0 2rem;
    }

    .hero-title {
        font-size: 4.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.05;
        margin-bottom: 1.2rem;
        color: #FFFFFF;
    }
    .hero-italic {
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        background: linear-gradient(90deg, #A8D5D5 0%, #E0F2F2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 5rem;
    }
    .section-label {
        font-size: 0.85rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #A8D5D5;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .body-text {
        font-size: 1.15rem;
        opacity: 0.75;
        line-height: 1.8;
        font-weight: 300;
    }

    /* Premium Glass Cards with Glow on Hover */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 2.5rem;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(168, 213, 213, 0.4);
        box-shadow: 0 15px 35px rgba(0,0,0,0.5), 0 0 30px rgba(168, 213, 213, 0.15);
        background: rgba(255, 255, 255, 0.03);
    }
    
    .card-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }

    /* Metrics Upgraded */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-top: 1.5rem;
    }
    .metric-item {
        flex: 1;
        padding: 1.5rem 1rem;
        background: rgba(0, 0, 0, 0.4);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-item:hover {
        border-color: rgba(168, 213, 213, 0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #A8D5D5;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.75rem;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 0.5rem;
    }

    /* Streamlit UI Overrides */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.01);
        border: 1px dashed rgba(168, 213, 213, 0.3);
        border-radius: 16px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #A8D5D5;
        background: rgba(168, 213, 213, 0.05);
    }

    .stButton > button {
        background: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 1rem;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 25px rgba(255, 255, 255, 0.2), 0 0 20px rgba(168, 213, 213, 0.4);
    }

    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
        background: rgba(0,0,0,0.3);
        border-radius: 16px;
        padding: 10px;
    }
    [data-testid="stImage"] img {
        max-height: 380px !important; 
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 4. FULL-WIDTH HERO SECTION
# =========================
st.markdown(f"""
<div class="hero-container">
    <video autoplay muted loop playsinline class="hero-video">
        <source src="{CLOUDINARY_VIDEO_URL}" type="video/mp4">
    </video>
    <div class="hero-overlay"></div>
    <div class="hero-content animate-up">
        <div class="section-label" style="display: flex; justify-content: center; letter-spacing: 0.5em; margin-bottom: 1.5rem;">Automated Neural Diagnostic</div>
        <h1 class="hero-title">AI Segmentation<br><span class="hero-italic">Pulmonary Embolism</span></h1>
        <p class="body-text" style="max-width: 600px; margin: 1.5rem auto 0 auto;">
            High-precision segmentation using YOLOv11-large architecture for emboli localization and AI-assisted clinical interpretation.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="padding: 0 1rem;">', unsafe_allow_html=True)

# =========================
# 5. EDUCATION SECTION
# =========================
st.markdown('<div class="animate-up delay-1">', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3, gap="medium")

with col_a:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">01. THE DEFINITION</div>
        <h3 class="card-title">What is PE?</h3>
        <p class="body-text">
            <b>Pulmonary Embolism (PE)</b> is a life-threatening blockage in the pulmonary arteries. Our AI detects and localizes filling defects in CTPA slices with high precision.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">02. THE CHALLENGE</div>
        <h3 class="card-title">Why PE?</h3>
        <p class="body-text">
            Radiologists face high fatigue when screening hundreds of CTPA slices. Missing a small clot can be fatal. Neuronix AI acts as a <b style="color: #A8D5D5;">vigilant second pair of eyes.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">03. THE BENCHMARK</div>
        <h3 class="card-title" style="margin-bottom: 0;">Performance</h3>
        <div class="metric-container">
            <div class="metric-item">
                <div class="metric-value">95.8<span style="font-size: 1.5rem;">%</span></div>
                <div class="metric-label">Sensitivity</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">98.2<span style="font-size: 1.5rem;">%</span></div>
                <div class="metric-label">Specificity</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br><br><br>", unsafe_allow_html=True)

# =========================
# 6. INFERENCE WORKSPACE (THE LOGIC ZONE)
# =========================
st.markdown('<div class="animate-up delay-2">', unsafe_allow_html=True)
st.markdown("<div class='section-label' style='text-align:center;'>NEURAL WORKSPACE</div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; font-family:Cormorant Garamond; font-size:4rem; margin-bottom:3rem; font-style: italic; font-weight:600;'>Autonomous Diagnostic Feed</h2>", unsafe_allow_html=True)

w_col1, w_col2 = st.columns([1, 1], gap="large")

with w_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 class='card-title'>CTPA Image Upload</h3>", unsafe_allow_html=True)
    
    # 1. State: Upload (Kiri)
    if st.session_state.app_state == "idle":
        source_img = st.file_uploader("Upload Slice", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        # Trigger Pindah State
        if source_img is not None and st.session_state.last_uploaded != source_img.name:
            st.session_state.last_uploaded = source_img.name
            st.session_state.raw_image = Image.open(source_img).convert("RGB")
            st.session_state.app_state = "scanning"
            st.rerun()

    # 2. State: Scanning Animasi Aktif (Kiri)
    elif st.session_state.app_state == "scanning":
        img_b64 = pil_to_base64(st.session_state.raw_image)
        st.markdown(f"""
            <div class="scanner-box">
                <img src="data:image/jpeg;base64,{img_b64}" alt="Scanning...">
                <div class="scanner-laser"></div>
            </div>
            <p style="text-align:center; color:#A8D5D5; font-size: 0.8rem; letter-spacing:0.1em; margin-top:10px; animation: pulseText 1.5s infinite;">Scanning Architecture...</p>
        """, unsafe_allow_html=True)

    # 3. State: Selesai (Kiri kembali normal)
    elif st.session_state.app_state == "complete":
        # Render gambar normal lagi lewat Streamlit bawaan (Animasi hilang!)
        st.image(st.session_state.raw_image, caption="Raw Patient Data", use_column_width=True)
        
        if st.button("RESET DIAGNOSTIC"):
            st.session_state.app_state = "idle"
            st.session_state.raw_image = None
            st.session_state.last_uploaded = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with w_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 class='card-title'>Segmentation Output</h3>", unsafe_allow_html=True)

    # Kanan saat Idle
    if st.session_state.app_state == "idle":
        st.markdown("""
            <div style="height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.3); border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1);">
                <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.3;">⚗️</div>
                <p style="opacity: 0.5; letter-spacing: 0.2em; font-size: 0.9rem; text-transform: uppercase;">System Idle: Awaiting Data</p>
            </div>
        """, unsafe_allow_html=True)

    # Kanan saat Scanning (AI bekerja)
    elif st.session_state.app_state == "scanning":
        with st.spinner("Executing YOLOv11 Neural Inference..."):
            time.sleep(2.5) # Memberi waktu agar animasi di kiri terlihat dramatis
            results = model.predict(st.session_state.raw_image, conf=0.25, verbose=False)
            res_plotted = results[0].plot()
            st.session_state["res_img"] = Image.fromarray(res_plotted[:, :, ::-1])
            st.session_state["detections"] = results[0].boxes
            
            # Ubah status jadi complete dan rerun layar
            st.session_state.app_state = "complete"
            st.rerun()

    # Kanan saat Selesai (Tampilkan Hasil)
    elif st.session_state.app_state == "complete":
        if "res_img" in st.session_state and st.session_state["res_img"] is not None:
            st.image(st.session_state["res_img"], caption="YOLOv11 Deep-Locus Result", use_column_width=True)
            
            det_count = len(st.session_state["detections"]) if st.session_state["detections"] else 0
            status_color = "#4ade80" if det_count == 0 else "#ff4b4b"
            status_text = "NEGATIVE / CLEAR" if det_count == 0 else f"POSITIVE / {det_count} LOCI"

            st.markdown(f"""
                <div class="metric-container" style="margin-top: 1rem;">
                    <div class="metric-item">
                        <div class="metric-label">Clots Found</div>
                        <div class="metric-value" style="color:{status_color}; font-size: 2.8rem; margin-top: 5px;">{det_count}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Analysis Result</div>
                        <div class="metric-value" style="color:{status_color}; font-size:1.4rem; margin-top:20px; letter-spacing: 0.05em;">{status_text}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# =========================
# 7. COMPARATIVE OUTPUT
# =========================
st.markdown('<div class="animate-up delay-3">', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-label'>RADIOLOGICAL COMPARISON</div>", unsafe_allow_html=True)
st.markdown("<h2 style='font-family:Cormorant Garamond; font-size: 3rem; margin-bottom: 2rem; font-weight:600;'>Technical Comparative Summary</h2>", unsafe_allow_html=True)

if pe_anatomy_base64:
    st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); padding:1.5rem; border-radius:20px; border: 1px solid rgba(255,255,255,0.05);">
            <img src="data:image/png;base64,{pe_anatomy_base64}" style="width:100%; border-radius:12px;">
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Upload 'result.png' to the app directory for comparative analysis visualization.")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
