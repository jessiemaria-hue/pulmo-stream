import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
from PIL import Image, ImageDraw
from pathlib import Path
import re
import io
import base64
import cv2
import numpy as np
# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PE Segmentation Review",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)
BASE_DIR = Path(__file__).resolve().parent
LABEL_DIR = BASE_DIR / "labels"
SHOW_DEBUG = False
# Uploaded trained segmentation checkpoints
YOLO11_PATH = BASE_DIR / "yolo11m.pt"
YOLO26_PATH = BASE_DIR / "yolo26l.pt"
CLOUDINARY_VIDEO_URL = (
    "https://res.cloudinary.com/dkxoksnpz/video/upload/"
    "f_mp4,vc_h264,q_auto:low,w_1280/"
    "v1774362450/3130284-uhd_3840_2160_30fps_mzryjc.mp4?v=3"
)
# =========================================================
# 2. LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    if not YOLO11_PATH.exists():
        raise FileNotFoundError(
            f"{YOLO11_PATH.name} not found. Put your trained segmentation best.pt beside app.py and update YOLO11_PATH."
        )
    if not YOLO26_PATH.exists():
        raise FileNotFoundError(f"{YOLO26_PATH.name} not found. Update YOLO26_PATH to your actual segmentation checkpoint.")
    model11 = YOLO(str(YOLO11_PATH))
    model26 = YOLO(str(YOLO26_PATH))
    for name, model in [("YOLOv11", model11), ("YOLOv26", model26)]:
        if getattr(model, "task", None) != "segment":
            raise ValueError(f"{name} checkpoint is a '{model.task}' model, NOT segmentation. Use trained -seg/best.pt weights.")
    return model11, model26
try:
    model11, model26 = load_models()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()
# =========================================================
# 3. HELPERS
# =========================================================
def normalize_stem(stem: str) -> str:
    clean = stem.strip()
    clean = re.sub(r"\s+copy(?:\s+\d+)?$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*\(\d+\)$", "", clean)
    return clean.strip()
def find_matching_label(image_filename: str):
    if not LABEL_DIR.exists():
        return None
    image_stem = Path(image_filename).stem
    exact = LABEL_DIR / f"{image_stem}.txt"
    if exact.exists():
        return exact
    normalized = normalize_stem(image_stem)
    normalized_path = LABEL_DIR / f"{normalized}.txt"
    if normalized_path.exists():
        return normalized_path
    target = normalized.lower()
    for txt_path in LABEL_DIR.glob("*.txt"):
        if txt_path.stem.lower() == target:
            return txt_path
    return None
def yolo_seg_txt_to_mask(txt_path: Path, image_size):
    """Build GT using the same cv2.fillPoly rasterization as Colab."""
    width, height = image_size
    merged = np.zeros((height, width), dtype=np.uint8)
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        except ValueError:
            continue
        if len(coords) % 2:
            continue
        points = coords.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(merged, [points], 1)
    return merged
def result_to_binary_mask(result, image_size):
    """Union native YOLO binary masks without discarding tiny polygons."""
    width, height = image_size
    merged = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None:
        return merged
    for tensor in result.masks.data:
        mask = tensor.detach().cpu().numpy()
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, (mask > 0.5).astype(np.uint8))
    return merged
def overlay_mask(raw_img, binary_mask, color, alpha=0.70):
    raw = np.asarray(raw_img.convert("RGB")).astype(np.float32)
    output = raw.copy()
    mask = binary_mask.astype(bool)
    if np.any(mask):
        c = np.array(color, dtype=np.float32)
        output[mask] = raw[mask] * (1 - alpha) + c * alpha
        edges, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, edges, -1, color, 2)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
def calculate_iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return None
    return float(intersection / union)
def get_best_confidence(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    conf = result.boxes.conf.detach().cpu().numpy()
    if len(conf) == 0:
        return None
    return float(np.max(conf))

def pil_to_data_uri(img, max_size=(560, 360)):
    preview = img.copy()
    preview.thumbnail(max_size)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# =========================================================
# 4. CSS
# =========================================================
_main_style = r"""



    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap" rel="stylesheet">



    <style>



        :root{



            --ink:#1F2A2B;



            --muted:#708084;



            --mint:#BFE7DB;



            --mint-strong:#5AAE95;



            --sage:#DDEEDC;



            --lav:#E8E2FF;



            --peach:#FFE6D8;



            --sky:#DDF1F7;



            --cream:#FFFDF8;



            --card:#FFFFFF;



            --line:#E6ECEB;



        }



        .stApp{



            background:



                radial-gradient(circle at 8% 10%, rgba(191,231,219,.38), transparent 24%),



                radial-gradient(circle at 92% 6%, rgba(232,226,255,.42), transparent 25%),



                radial-gradient(circle at 86% 82%, rgba(255,230,216,.30), transparent 24%),



                linear-gradient(180deg,#FFFDF8 0%,#FAFCFB 48%,#F6FAF8 100%);



            color:var(--ink);



            font-family:'Source Sans 3',Arial,sans-serif;



        }



        .block-container{



            max-width:1880px !important;



            width:98% !important;



            padding:0 1.4rem 4rem 1.4rem !important;



            margin:0 auto !important;



        }



        header,footer{visibility:hidden;}



        @keyframes floatUp{



            from{opacity:0;transform:translateY(24px);}



            to{opacity:1;transform:translateY(0);}



        }



        @keyframes breathe{



            0%,100%{transform:scale(1);}



            50%{transform:scale(1.015);}



        }



        @keyframes blobFloat{



            0%,100%{transform:translateY(0) rotate(0deg);}



            50%{transform:translateY(-10px) rotate(2deg);}



        }



        .hero-container{



            position:relative;



            width:100%;



            height:62vh;



            min-height:520px;



            display:flex;



            align-items:center;



            justify-content:center;



            margin-bottom:2.8rem;



            overflow:hidden;



            border-radius:0 0 42px 42px;



            border:1px solid #E5ECEA;



            border-top:none;



            background:#EEF4F1;



            box-shadow:0 24px 65px rgba(62,89,82,.12);



            animation:floatUp .8s ease both;



        }



        .hero-video{



            position:absolute;inset:0;width:100%;height:100%;object-fit:cover;



            opacity:.36;



            filter:contrast(1.04) brightness(1.10) saturate(.82);



            pointer-events:none;



        }



        .hero-overlay{



            position:absolute;inset:0;



            background:



                linear-gradient(90deg,rgba(255,253,248,.78),rgba(255,253,248,.42) 50%,rgba(255,253,248,.72)),



                linear-gradient(180deg,rgba(255,255,255,.08),rgba(255,255,255,.28) 58%,#FFFDF8 100%);



        }



        .hero-grid{display:none;}



        .hero-content{



            position:relative;



            z-index:2;



            width:min(980px,90%);



            text-align:center;



            padding:2rem;



            border-radius:32px;



            background:rgba(255,255,255,.58);



            border:1px solid rgba(255,255,255,.75);



            backdrop-filter:blur(14px);



            box-shadow:0 18px 46px rgba(71,98,91,.10);



        }



        .hero-kicker{



            display:inline-flex;



            align-items:center;



            gap:.55rem;



            border-radius:999px;



            padding:.52rem .82rem;



            background:var(--mint);



            color:#2F6B59;



            border:none;



            font-size:.70rem;



            font-weight:800;



            letter-spacing:.16em;



            text-transform:uppercase;



            margin-bottom:1rem;



        }



        .hero-kicker:before{



            content:"";



            width:8px;height:8px;border-radius:50%;



            background:#4DA98D;



        }



        .hero-title{



            color:#243133 !important;



            font-size:clamp(3.8rem,6vw,6.6rem);



            font-weight:800;



            line-height:.95;



            letter-spacing:-.05em;



            margin:0;



            text-shadow:none;



        }



        .hero-italic{



            display:inline-block;



            margin-top:.25rem;



            font-family:'Libre Baskerville',Georgia,serif;



            font-style:italic;



            color:#4A907B !important;



            background:none !important;



            -webkit-text-fill-color:#4A907B !important;



        }



        .hero-subtitle{



            max-width:760px;



            margin:1.35rem auto 0;



            color:#56686B !important;



            font-size:1rem;



            line-height:1.7;



            font-weight:500;



        }



        .hero-tags{



            display:flex;justify-content:center;gap:.55rem;flex-wrap:wrap;margin-top:1.35rem;



        }



        .hero-tag{



            border:none;



            border-radius:999px;



            padding:.46rem .72rem;



            font-size:.70rem;



            font-weight:700;



            color:#425557;



            background:#FFFFFF;



            box-shadow:0 6px 18px rgba(71,98,91,.07);



        }



        .hero-tag:nth-child(1){background:var(--sky);}



        .hero-tag:nth-child(2){background:var(--mint);}



        .hero-tag:nth-child(3){background:var(--sage);}



        .hero-tag:nth-child(4){background:var(--peach);}



        .hero-tag:nth-child(5){background:var(--lav);}



        .section-wrap{animation:floatUp .8s .08s ease both;}



        .section-head{



            display:flex;



            justify-content:space-between;



            align-items:end;



            gap:2rem;



            margin-bottom:1.15rem;



        }



        .section-label{



            font-size:.72rem;



            letter-spacing:.28em;



            text-transform:uppercase;



            color:#72A99A;



            font-weight:800;



            margin-bottom:.45rem;



        }



        .section-title{



            font-family:'Libre Baskerville',Georgia,serif;



            font-size:3.1rem;



            font-style:italic;



            font-weight:400;



            line-height:1;



            color:#243133;



            margin:0;



        }



        .section-copy{



            max-width:720px;



            color:#738286;



            font-size:.90rem;



            line-height:1.7;



            margin-top:.65rem;



        }



        .workflow{



            display:flex;gap:.4rem;align-items:center;white-space:nowrap;



            color:#7D8B8E;font-size:.65rem;font-weight:800;text-transform:uppercase;



        }



        .workflow span{



            padding:.40rem .60rem;



            border-radius:999px;



            background:#FFFFFF;



            border:1px solid var(--line);



            box-shadow:0 5px 14px rgba(71,98,91,.05);



        }



        .upload-card{



            position:relative;



            background:#FFFFFF;



            border:1px solid #E7ECEA;



            border-radius:24px;



            padding:.95rem;



            margin-bottom:1.25rem;



            box-shadow:0 16px 40px rgba(71,98,91,.08);



            overflow:hidden;



        }



        .upload-card:before{



            content:"";



            position:absolute;



            width:160px;height:160px;border-radius:50%;



            right:-80px;top:-90px;



            background:var(--lav);



            opacity:.55;



        }



        [data-testid="stFileUploader"]{



            position:relative;



            z-index:2;



            background:#FCFEFD;



            border:1px dashed #BFD7CF;



            border-radius:18px;



            padding:.70rem;



        }



        [data-testid="stFileUploader"] section{



            background:linear-gradient(90deg,#F6FCF9,#FBF9FF) !important;



            border-radius:14px !important;



        }



        [data-testid="stFileUploader"] *{



            color:#334244 !important;



        }



        .image-label{



            display:flex;



            justify-content:center;



            align-items:center;



            gap:.42rem;



            font-size:.84rem;



            font-weight:800;



            margin-bottom:.58rem;



            color:#2B393B;



        }



        .image-label .dot{



            width:8px;height:8px;border-radius:50%;display:inline-block;



        }



        .dot-raw{background:#A8B2B4;}



        .dot-gt{background:#62C58A;}



        .dot-11{background:#F08080;}



        .dot-26{background:#53BDD0;}



        [data-testid="stImage"]{



            background:#FFFFFF;



            border:1px solid #E5EBEA;



            border-radius:22px;



            padding:6px;



            box-shadow:0 14px 32px rgba(71,98,91,.09);



            transition:transform .25s ease,box-shadow .25s ease;



        }



        [data-testid="stImage"]:hover{



            transform:translateY(-4px);



            box-shadow:0 20px 44px rgba(71,98,91,.14);



        }



        [data-testid="stImage"] img{



            width:100% !important;



            height:auto !important;



            max-height:345px !important;



            object-fit:contain !important;



            border-radius:16px;



        }



        .info-card{



            position:relative;



            background:#FFFFFF;



            border:1px solid #E6ECEB;



            border-radius:22px;



            padding:.9rem;



            margin-bottom:.72rem;



            box-shadow:0 14px 32px rgba(71,98,91,.08);



            overflow:hidden;



        }



        .info-card:nth-of-type(1){background:linear-gradient(180deg,#FFFFFF,#F8FCFA);}



        .info-card:nth-of-type(2){background:linear-gradient(180deg,#FFFFFF,#FBFAFF);}



        .info-card:before{



            content:"";



            position:absolute;left:0;top:0;width:5px;height:100%;



            background:linear-gradient(180deg,#9CD8C5,#D5C9FF);



        }



        .model-head{



            display:flex;



            justify-content:space-between;



            align-items:center;



            margin-bottom:.62rem;



        }



        .model-badge{



            display:inline-flex;



            align-items:center;



            border:none;



            background:var(--mint);



            color:#2E6A58;



            border-radius:999px;



            padding:.38rem .66rem;



            font-size:.70rem;



            font-weight:800;



        }



        .model-status{



            color:#8A979A;



            font-size:.58rem;



            text-transform:uppercase;



            letter-spacing:.10em;



            font-weight:800;



        }



        .mini-grid{



            display:grid;



            grid-template-columns:1fr 1fr;



            gap:8px;



        }



        .mini-grid .mini-metric:last-child{



            grid-column:1 / -1;



        }



        .mini-metric{



            min-height:64px;



            background:#F8FBFA;



            border:1px solid #E9EFED;



            border-radius:14px;



            padding:.64rem;



        }



        .mini-value{



            font-size:1rem;



            font-weight:800;



            color:#376F60;



            line-height:1.1;



        }



        .mini-label{



            margin-top:.30rem;



            font-size:.58rem;



            color:#859296;



            text-transform:uppercase;



            letter-spacing:.09em;



            font-weight:800;



        }



        .legend-card{



            display:grid;



            grid-template-columns:1fr 1fr;



            gap:7px;



            margin-top:.78rem;



            background:#FFFFFF;



            border:1px solid #E7ECEB;



            border-radius:16px;



            padding:.70rem;



        }



        .legend-item{



            display:flex;



            align-items:center;



            gap:.4rem;



            color:#79878A;



            font-size:.64rem;



        }



        .legend-dot{



            width:8px;height:8px;border-radius:50%;



        }



        .explain-strip{



            margin-top:1.2rem;



            display:grid;



            grid-template-columns:repeat(2,1fr);



            gap:10px;



            animation:floatUp .8s .16s ease both;



        }



        .explain-item{



            border:none;



            border-radius:20px;



            padding:1rem;



            box-shadow:0 12px 26px rgba(71,98,91,.06);



        }



        .explain-item:nth-child(1){background:linear-gradient(135deg,#F2FBF6,#E6F5EF);}



        .explain-item:nth-child(2){background:linear-gradient(135deg,#FFF8F3,#FFEFE5);}



        .explain-item:nth-child(3){background:linear-gradient(135deg,#F8F6FF,#EDE9FF);}



        .explain-item b{



            display:block;



            color:#314345;



            font-size:.74rem;



            margin-bottom:.20rem;



        }



        .explain-item span{



            color:#708084;



            font-size:.69rem;



            line-height:1.5;



        }



        .stAlert{



            background:#F5FBF8 !important;



            border:1px solid #DDEBE5 !important;



            color:#37504A !important;



            border-radius:16px !important;



        }



        @keyframes slideInRight{



            from{opacity:0;transform:translateX(34px);}



            to{opacity:1;transform:translateX(0);}



        }



        @keyframes popIn{



            0%{opacity:0;transform:scale(.94) translateY(10px);}



            100%{opacity:1;transform:scale(1) translateY(0);}



        }



        @keyframes floatBlobA{



            0%,100%{transform:translate3d(0,0,0) scale(1);}



            50%{transform:translate3d(18px,-16px,0) scale(1.05);}



        }



        @keyframes floatBlobB{



            0%,100%{transform:translate3d(0,0,0) scale(1);}



            50%{transform:translate3d(-16px,14px,0) scale(.96);}



        }



        .hero-blob{



            position:absolute;



            border-radius:50%;



            filter:blur(4px);



            opacity:.38;



            pointer-events:none;



            z-index:1;



        }



        .blob-a{



            width:220px;



            height:220px;



            background:#DDF1F7;



            top:8%;



            left:6%;



            animation:floatBlobA 9s ease-in-out infinite;



        }



        .blob-b{



            width:260px;



            height:260px;



            background:#E8E2FF;



            bottom:4%;



            right:7%;



            animation:floatBlobB 11s ease-in-out infinite;



        }



        .stagger-1{animation:popIn .55s .05s ease both;}



        .stagger-2{animation:popIn .55s .14s ease both;}



        .stagger-3{animation:popIn .55s .23s ease both;}



        .stagger-4{animation:popIn .55s .32s ease both;}



        .stagger-5{animation:popIn .55s .41s ease both;}



        .slide-right{



            animation:slideInRight .7s .18s cubic-bezier(.16,1,.3,1) both;



        }



        .metric-pop{



            animation:popIn .55s ease both;



        }



        .scroll-reveal{



            opacity:0;



            transform:translateY(24px);



            transition:



                opacity .72s cubic-bezier(.16,1,.3,1),



                transform .72s cubic-bezier(.16,1,.3,1);



            will-change:opacity, transform;



        }



        .scroll-reveal.visible{



            opacity:1;



            transform:translateY(0);



        }



        .scroll-reveal-delay-1{transition-delay:.08s;}



        .scroll-reveal-delay-2{transition-delay:.16s;}



        .scroll-reveal-delay-3{transition-delay:.24s;}



        .scroll-reveal-delay-4{transition-delay:.32s;}



        /* ---------- Gen-Z health-tech refinements ---------- */



        [data-testid="stFileUploader"]{



            background:linear-gradient(135deg,#F8FCFA 0%,#FBFAFF 100%) !important;



            border:1.5px dashed #B6D7CD !important;



            box-shadow:0 12px 28px rgba(73,112,101,.07);



        }



        [data-testid="stFileUploader"] section{



            background:transparent !important;



        }



        [data-testid="stFileUploader"] button{



            background:#315F63 !important;



            color:#FFFFFF !important;



            border:0 !important;



            border-radius:12px !important;



            padding:.55rem .85rem !important;



            box-shadow:0 8px 20px rgba(49,95,99,.18) !important;



            transition:transform .22s ease, box-shadow .22s ease, background .22s ease !important;



        }



        [data-testid="stFileUploader"] button:hover{



            background:#274E52 !important;



            transform:translateY(-1px);



            box-shadow:0 11px 24px rgba(49,95,99,.24) !important;



        }



        [data-testid="stFileUploader"] button *{



            color:#FFFFFF !important;



        }



        [data-testid="stFileUploader"] small,

        [data-testid="stFileUploader"] p{

            color:#405155 !important;



        }

        [data-testid="stFileUploader"] span,

            color:#FFFFFF !important;


        .mini-feature-row{



            display:grid;



            grid-template-columns:repeat(3,1fr);



            gap:10px;



            margin:0 0 1.1rem 0;



        }



        .mini-feature{



            position:relative;



            overflow:hidden;



            min-height:84px;



            border-radius:20px;



            padding:1rem 1.05rem;



            border:1px solid #E5ECE9;



            box-shadow:0 12px 28px rgba(71,98,91,.06);



        }



        .mini-feature:nth-child(1){



            background:linear-gradient(135deg,#F2FBF6,#E7F6EF);



        }



        .mini-feature:nth-child(2){



            background:linear-gradient(135deg,#FFF8F3,#FFEDE1);



        }



        .mini-feature:nth-child(3){



            background:linear-gradient(135deg,#F7F5FF,#ECE8FF);



        }



        .mini-feature:after{



            content:"";



            position:absolute;



            width:90px;height:90px;border-radius:50%;



            right:-35px;top:-40px;



            background:rgba(255,255,255,.55);



        }



        .mini-feature-top{



            display:flex;



            align-items:center;



            gap:.55rem;



            margin-bottom:.35rem;



        }



        .mini-icon{



            width:27px;height:27px;border-radius:9px;



            display:flex;align-items:center;justify-content:center;



            font-size:.82rem;font-weight:800;



            background:rgba(255,255,255,.82);



            color:#3C6F61;



            box-shadow:0 5px 12px rgba(71,98,91,.06);



        }



        .mini-feature b{



            font-size:.78rem;



            color:#314143;



        }



        .mini-feature p{



            margin:0;



            font-size:.68rem;



            line-height:1.45;



            color:#748286;



        }



        .benchmark-shell{



            position:relative;



            margin-top:.45rem;



            padding:1rem .85rem .9rem;



            border-radius:24px;



            border:1px solid rgba(220,231,227,.95);



            background:rgba(255,255,255,.50);



            box-shadow:0 18px 42px rgba(71,98,91,.055);



        }



        .benchmark-shell:before{



            content:"Benchmark View";



            position:absolute;



            top:-11px;



            left:18px;



            background:#FFFFFF;



            border:1px solid #E1EAE7;



            color:#75928A;



            border-radius:999px;



            padding:.24rem .56rem;



            font-size:.56rem;



            font-weight:800;



            letter-spacing:.10em;



            text-transform:uppercase;



        }



        @media(max-width:1000px){



            .mini-feature-row{grid-template-columns:1fr;}



        }



        /* ---------- Stronger product UI ---------- */



        html, body, .stApp, button, input, textarea, select, a {



            cursor:



              url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Ccircle cx='8' cy='8' r='5' fill='%235AAE95' fill-opacity='.95'/%3E%3Ccircle cx='8' cy='8' r='7.2' fill='none' stroke='%23FFFFFF' stroke-width='1.5'/%3E%3C/svg%3E") 8 8,



              auto;



        }



        button, a, [role="button"], [data-testid="stFileUploader"] button {



            cursor:



              url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 28 28'%3E%3Ccircle cx='9' cy='9' r='6' fill='%234A907B'/%3E%3Ccircle cx='9' cy='9' r='8.5' fill='none' stroke='%23FFFFFF' stroke-width='1.8'/%3E%3C/svg%3E") 9 9,



              pointer;



        }



        .topbar{



            position:relative;



            z-index:20;



            display:flex;



            justify-content:space-between;



            align-items:center;



            gap:1rem;



            margin:.85rem 0 .9rem;



            padding:.72rem .9rem;



            border:1px solid rgba(225,235,232,.95);



            border-radius:18px;



            background:rgba(255,255,255,.78);



            backdrop-filter:blur(16px);



            box-shadow:0 12px 28px rgba(71,98,91,.07);



            animation:floatUp .65s ease both;



        }



        .brand-wrap{



            display:flex;



            align-items:center;



            gap:.7rem;



        }



        .brand-mark{



            width:34px;



            height:34px;



            border-radius:11px;



            display:flex;



            align-items:center;



            justify-content:center;



            font-weight:900;



            font-size:.82rem;



            color:#FFFFFF;



            background:linear-gradient(135deg,#5AAE95,#7EC9B1);



            box-shadow:0 8px 20px rgba(90,174,149,.20);



        }



        .brand-text{



            line-height:1.05;



        }



        .brand-text b{



            display:block;



            color:#263638;



            font-size:.80rem;



            letter-spacing:.01em;



        }



        .brand-text span{



            display:block;



            color:#8A989B;



            font-size:.58rem;



            margin-top:.12rem;



            letter-spacing:.06em;



            text-transform:uppercase;



            font-weight:700;



        }



        .topbar-right{



            display:flex;



            align-items:center;



            gap:.5rem;



            flex-wrap:wrap;



            justify-content:flex-end;



        }



        .top-chip{



            display:flex;



            align-items:center;



            gap:.38rem;



            padding:.38rem .56rem;



            border-radius:999px;



            background:#F7FAF9;



            border:1px solid #E5ECEA;



            color:#6C7B7E;



            font-size:.60rem;



            font-weight:800;



            letter-spacing:.04em;



            text-transform:uppercase;



        }



        .live-dot{



            width:7px;



            height:7px;



            border-radius:50%;



            background:#61C38B;



            box-shadow:0 0 0 5px rgba(97,195,139,.10);



            animation:breathe 2.4s ease-in-out infinite;



        }



        .section-head{



            position:relative;



            padding:1.2rem 1.25rem;



            border:1px solid #E7ECEA;



            border-radius:24px;



            background:



                radial-gradient(circle at 5% 100%,rgba(191,231,219,.30),transparent 26%),



                rgba(255,255,255,.92);



            box-shadow:0 14px 32px rgba(71,98,91,.06);



        }



        .section-head:after{



            content:"";



            position:absolute;



            inset:auto 16px -1px 16px;



            height:2px;



            border-radius:999px;



            background:linear-gradient(90deg,#BFE7DB,#D6EEE6,#E7F5F0);



            opacity:.95;



        }



        .upload-card{



            transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;



        }



        .upload-card:hover{



            transform:translateY(-3px);



            border-color:#CFE2DB;



            box-shadow:0 20px 44px rgba(71,98,91,.11);



        }



        .benchmark-shell{



            background:



                radial-gradient(circle at 0% 0%,rgba(191,231,219,.18),transparent 22%),



                rgba(255,255,255,.72);



            border:1px solid #E3ECE8;



            box-shadow:0 18px 46px rgba(71,98,91,.075);



        }



        .benchmark-shell:after{



            content:"";



            position:absolute;



            left:18px;



            right:18px;



            bottom:8px;



            height:1px;



            background:linear-gradient(90deg,transparent,#D4E6E0,transparent);



        }



        .image-label{



            position:relative;



            padding:.42rem .55rem;



            width:max-content;



            margin-left:auto;



            margin-right:auto;



            border-radius:999px;



            background:rgba(255,255,255,.80);



            border:1px solid #E7EEEC;



            box-shadow:0 6px 16px rgba(71,98,91,.05);



        }



        [data-testid="stImage"]{



            position:relative;



            overflow:hidden;



        }



        [data-testid="stImage"]:after{



            content:"";



            position:absolute;



            inset:6px;



            border-radius:16px;



            pointer-events:none;



            background:linear-gradient(



                110deg,



                transparent 0%,



                transparent 42%,



                rgba(255,255,255,.20) 50%,



                transparent 58%,



                transparent 100%



            );



            transform:translateX(-130%);



            transition:transform .7s ease;



        }



        [data-testid="stImage"]:hover:after{



            transform:translateX(130%);



        }



        .info-card{



            border-top:1px solid rgba(255,255,255,.95);



        }



        .info-card:hover{



            transform:translateY(-4px) scale(1.01);



            box-shadow:0 18px 40px rgba(71,98,91,.12);



        }



        .mini-metric{



            transition:transform .20s ease, background .20s ease, border-color .20s ease;



        }



        .mini-metric:hover{



            transform:translateY(-2px);



            background:#FFFFFF;



            border-color:#D5E4DF;



        }



        .result-ribbon{



            display:flex;



            align-items:center;



            justify-content:space-between;



            gap:1rem;



            margin:1rem 0 .75rem;



            padding:.72rem .9rem;



            border-radius:18px;



            background:linear-gradient(90deg,#F3FBF7,#FFF9F5 52%,#F8F6FF);



            border:1px solid #E6ECEA;



            box-shadow:0 10px 24px rgba(71,98,91,.05);



        }



        .result-ribbon-left{



            display:flex;



            align-items:center;



            gap:.55rem;



        }



        .result-ribbon-icon{



            width:30px;



            height:30px;



            border-radius:10px;



            background:#BFE7DB;



            color:#2F6B59;



            display:flex;



            align-items:center;



            justify-content:center;



            font-weight:900;



            font-size:.72rem;



        }



        .result-ribbon-text b{



            display:block;



            color:#314244;



            font-size:.74rem;



        }



        .result-ribbon-text span{



            color:#7C8A8D;



            font-size:.63rem;



        }



        .result-ribbon-right{



            display:flex;



            align-items:center;



            gap:.4rem;



            color:#7D8A8D;



            font-size:.61rem;



            font-weight:800;



            text-transform:uppercase;



            letter-spacing:.04em;



        }



        .result-ribbon-right span{



            padding:.34rem .5rem;



            border-radius:999px;



            background:#FFFFFF;



            border:1px solid #E6ECEA;



        }



        @media(max-width:1100px){



            .workflow{display:none;}



            .hero-container{height:58vh;min-height:480px;}



            .hero-content{width:92%;}



            .explain-strip{grid-template-columns:1fr;}



        }



        /* Keep the benchmark heading and uploader close together */



        .section-head { margin-bottom: .35rem !important; }



        .section-wrap { margin-bottom: 0 !important; }



        [data-testid="stFileUploader"] { margin-top: 0 !important; }



        .upload-card { margin-top: 0 !important; }







        /* Compact benchmark header and uploader layout */



        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {



            gap: .65rem !important;



        }



        .section-wrap { margin: 0 !important; }



        .section-head { margin-bottom: 0 !important; }



        .upload-card { margin-top: 0 !important; margin-bottom: .4rem !important; }



        [data-testid="stFileUploader"] { margin: 0 !important; }



        [data-testid="stFileUploader"] section { min-height: 68px !important; }



        [data-testid="stFileUploader"]{



            padding: 1rem !important;



            border-radius:24px !important;



            border:1px dashed #BFD7CF !important;



            background:linear-gradient(180deg,#FCFEFD 0%,#FAFCFB 100%) !important;



            box-shadow:0 12px 28px rgba(71,98,91,.05) !important;



        }







        /* ---------- Academic + AI attribution ---------- */



        .credit-strip{display:none !important;



            display:flex;



            align-items:center;



            justify-content:center;



            gap:.55rem;



            flex-wrap:wrap;



            margin:-.25rem 0 .9rem 0;



        }



        .credit-chip{



            display:inline-flex;



            align-items:center;



            gap:.48rem;



            padding:.42rem .66rem;



            border-radius:999px;



            border:1px solid #E2EAE7;



            background:rgba(255,255,255,.86);



            box-shadow:0 7px 18px rgba(71,98,91,.055);



            color:#617376;



            font-size:.60rem;



            line-height:1;



            font-weight:700;



            letter-spacing:.015em;



        }



        .credit-chip strong{



            color:#344749;



            font-weight:800;



        }



        .credit-mark{



            width:25px;



            height:25px;



            border-radius:8px;



            display:inline-flex;



            align-items:center;



            justify-content:center;



            flex:0 0 25px;



            font-size:.62rem;



            font-weight:900;



        }



        .credit-mark-i3l{



            background:linear-gradient(135deg,#EAF7F2,#D8EFE7);



            color:#397261;



            border:1px solid #CFE6DE;



            letter-spacing:-.04em;



        }



        .credit-mark-gpt{



            position:relative;



            background:#EEF5F3;



            border:1px solid #D8E5E1;



        }



        .credit-mark-gpt:before{



            content:"✦";



            color:#3F7063;



            font-size:.88rem;



            line-height:1;



        }



        .credit-sep{



            color:#A8B4B2;



            font-size:.62rem;



            font-weight:700;



        }



        @media(max-width:700px){



            .credit-strip{display:none !important;gap:.4rem;}



            .credit-chip{font-size:.56rem;padding:.38rem .58rem;}



        }








        /* =========================================================
           FINAL MEDICAL EDITORIAL OVERRIDES
           ========================================================= */
        :root{
            --med-ink:#172526;
            --med-muted:#66787A;
            --med-teal:#2E6A61;
            --med-teal-soft:#DCEBE6;
            --med-line:#D9E3E0;
            --med-paper:#F7F9F8;
            --med-white:#FFFFFF;
        }

        .stApp{
            background:
                radial-gradient(circle at 8% 2%, rgba(207,229,221,.34), transparent 24%),
                linear-gradient(180deg,#F8FAF9 0%,#F4F7F6 100%) !important;
            color:var(--med-ink) !important;
        }

        .block-container{
            max-width:1840px !important;
            width:97% !important;
            padding:0 1.25rem 3.5rem !important;
        }

        /* Remove the old pastel/product feel */
        .hero-blob,.blob-a,.blob-b{display:none !important;}

        .section-head{
            border:none !important;
            border-radius:0 !important;
            background:transparent !important;
            box-shadow:none !important;
            padding:.65rem .10rem .85rem !important;
            margin-bottom:.15rem !important;
        }
        .section-head:after{display:none !important;}

        .section-label{
            color:#5D9385 !important;
            font-size:.64rem !important;
            letter-spacing:.31em !important;
            font-weight:700 !important;
        }

        .section-title{
            color:#172526 !important;
            font-family:'Libre Baskerville',Georgia,serif !important;
            font-style:normal !important;
            font-size:3rem !important;
            letter-spacing:-.025em !important;
        }

        .section-copy{
            color:#68797B !important;
            font-size:.86rem !important;
            max-width:760px !important;
            line-height:1.65 !important;
        }

        .workflow{
            color:#7A898B !important;
            gap:.35rem !important;
        }
        .workflow span{
            background:rgba(255,255,255,.88) !important;
            border:1px solid #DFE7E5 !important;
            box-shadow:none !important;
            padding:.46rem .68rem !important;
            color:#67787A !important;
        }
        .workflow span:first-child{
            color:#2D6C61 !important;
            background:#E8F3EF !important;
            border-color:#D5E7E1 !important;
        }

        [data-testid="stFileUploader"]{
            background:#FFFFFF !important;
            border:1px dashed #AFCFC4 !important;
            border-radius:20px !important;
            box-shadow:0 12px 30px rgba(38,67,61,.045) !important;
            padding:.82rem !important;
        }
        [data-testid="stFileUploader"] section{
            background:#FFFFFF !important;
            border-radius:16px !important;
            min-height:78px !important;
        }
        [data-testid="stFileUploader"] button{
            background:#245B56 !important;
            border-radius:11px !important;
            box-shadow:none !important;
            font-weight:650 !important;
        }

        .result-ribbon{
            background:#F0F6F3 !important;
            border:1px solid #DCE7E3 !important;
            border-radius:16px !important;
            box-shadow:none !important;
        }
        .result-ribbon-icon{
            background:#D7EAE3 !important;
            color:#285F55 !important;
        }

        .benchmark-shell{
            background:rgba(255,255,255,.78) !important;
            border:1px solid #DEE7E4 !important;
            box-shadow:0 16px 40px rgba(38,67,61,.05) !important;
            border-radius:22px !important;
        }
        .benchmark-shell:before{
            color:#69817B !important;
            border-color:#DEE7E4 !important;
            box-shadow:none !important;
        }

        .image-label{
            box-shadow:none !important;
            border:1px solid #E0E8E6 !important;
            background:#FBFCFC !important;
            color:#2B3A3C !important;
        }

        [data-testid="stImage"]{
            border:1px solid #DFE7E5 !important;
            border-radius:18px !important;
            box-shadow:0 10px 26px rgba(36,63,58,.055) !important;
        }
        [data-testid="stImage"]:hover{
            transform:none !important;
            box-shadow:0 12px 30px rgba(36,63,58,.075) !important;
        }
        [data-testid="stImage"]:after{display:none !important;}

        .info-card{
            border:1px solid #DFE7E5 !important;
            border-radius:18px !important;
            background:#FFFFFF !important;
            box-shadow:0 10px 26px rgba(36,63,58,.05) !important;
        }
        .info-card:hover{
            transform:none !important;
            box-shadow:0 10px 26px rgba(36,63,58,.05) !important;
        }
        .info-card:before{
            width:3px !important;
            background:#7DB7A6 !important;
        }
        .model-badge{
            background:#E1EFEA !important;
            color:#2A655A !important;
        }
        .mini-metric{
            background:#F7FAF9 !important;
            border-color:#E4ECE9 !important;
        }
        .mini-value{color:#285E55 !important;}

        .legend-card{
            box-shadow:none !important;
            border-color:#E0E8E6 !important;
        }

        .explain-strip{gap:12px !important;}
        .explain-item{
            background:#FFFFFF !important;
            border:1px solid #E0E8E6 !important;
            box-shadow:none !important;
            border-radius:16px !important;
        }

        /* Real brand / attribution placement */
        .research-credit{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:1rem;
            margin:1.15rem .10rem .25rem;
            padding:.75rem .85rem;
            border-top:1px solid #DFE7E4;
            color:#839092;
        }
        .research-credit-left,
        .research-credit-right{
            display:flex;
            align-items:center;
            gap:.58rem;
        }
        .research-credit .i3l-logo{
            width:42px;
            height:42px;
            object-fit:contain;
            border-radius:8px;
        }
        .research-credit .gpt-logo{
            width:27px;
            height:27px;
            object-fit:contain;
        }
        .research-credit strong{
            display:block;
            color:#4B5E60;
            font-size:.63rem;
            font-weight:650;
            line-height:1.2;
        }
        .research-credit span{
            display:block;
            margin-top:.08rem;
            font-size:.54rem;
            color:#899698;
            line-height:1.25;
        }

        @media(max-width:800px){
            .research-credit{align-items:flex-start;flex-direction:column;}
            .section-title{font-size:2.35rem !important;}
        }


/* ===== CLINICAL EDITORIAL + YOUTHFUL POLISH ===== */
.stApp{
    background:
        radial-gradient(circle at 8% 6%, rgba(196,224,215,.24), transparent 24%),
        linear-gradient(180deg,#F8FAF9 0%,#F5F8F7 100%) !important;
}
.clinical-flow{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin:.9rem 0 1.2rem;
}
.clinical-step{
    position:relative;
    padding:.9rem .95rem .95rem;
    border-radius:16px;
    background:#FFFFFF;
    border:1px solid #E0E8E5;
    box-shadow:0 8px 22px rgba(38,66,60,.04);
}
.clinical-step:before{
    content:"";
    position:absolute;
    left:0;top:0;bottom:0;width:3px;
    border-radius:16px 0 0 16px;
    background:#77A99A;
}
.clinical-step:nth-child(2):before{background:#A6B886}
.clinical-step:nth-child(3):before{background:#D7A487}
.clinical-step:nth-child(4):before{background:#89AFC0}
.clinical-index{
    font-size:.53rem;
    letter-spacing:.16em;
    text-transform:uppercase;
    color:#93A09F;
    font-weight:800;
    margin-bottom:.28rem;
}
.clinical-step b{
    display:block;
    color:#2D3E40;
    font-size:.73rem;
    margin-bottom:.16rem;
}
.clinical-step span{
    display:block;
    color:#798789;
    font-size:.61rem;
    line-height:1.45;
}
.study-row{
    display:grid;
    grid-template-columns:1.05fr 1fr 1fr;
    gap:12px;
    margin:.15rem 0 1rem;
}
.study-card{
    min-height:92px;
    padding:.95rem 1rem;
    border-radius:16px;
    border:1px solid #E0E8E5;
    background:#FFFFFF;
}
.study-card.featured{
    background:linear-gradient(135deg,#EEF7F3,#F7FAF8);
    border-color:#D5E7E1;
}
.study-label{
    font-size:.53rem;
    letter-spacing:.14em;
    text-transform:uppercase;
    font-weight:800;
    color:#81908F;
    margin-bottom:.30rem;
}
.study-card b{
    display:block;
    color:#2D3F41;
    font-size:.76rem;
    margin-bottom:.22rem;
}
.study-card p{
    margin:0;
    color:#768588;
    font-size:.63rem;
    line-height:1.48;
}
.brand-note{
    display:flex;
    justify-content:flex-end;
    align-items:center;
    gap:.42rem;
    margin:.15rem .10rem .8rem;
    color:#849193;
    font-size:.56rem;
    font-weight:600;
}
.brand-note img{
    width:22px;height:22px;object-fit:contain;
}
@media(max-width:900px){
    .clinical-flow{grid-template-columns:1fr 1fr;}
    .study-row{grid-template-columns:1fr;}
}
@media(max-width:620px){
    .clinical-flow{grid-template-columns:1fr;}
}


/* ===== REFINED MEDICAL / GEN-Z DETAILS ===== */
.review-path{
    margin:.95rem 0 1.15rem;
    padding:.95rem 1.05rem 1rem;
    border-radius:18px;
    background:
        linear-gradient(180deg,rgba(255,255,255,.94),rgba(249,252,251,.94));
    border:1px solid #DEE7E4;
    box-shadow:0 10px 28px rgba(37,64,59,.045);
}
.review-path-title{
    display:flex;
    align-items:baseline;
    justify-content:space-between;
    gap:1rem;
    margin-bottom:.85rem;
}
.review-path-title span{
    color:#2D4243;
    font-size:.70rem;
    font-weight:800;
    letter-spacing:.12em;
    text-transform:uppercase;
}
.review-path-title small{
    color:#8A989A;
    font-size:.58rem;
    font-weight:600;
}
.review-path-track{
    display:flex;
    align-items:center;
    gap:.55rem;
}
.review-node{
    display:flex;
    align-items:center;
    gap:.55rem;
    min-width:0;
    flex:1;
}
.review-node b{
    display:block;
    color:#304244;
    font-size:.69rem;
    line-height:1.2;
}
.review-node small{
    display:block;
    color:#8A9799;
    font-size:.56rem;
    margin-top:.12rem;
}
.review-dot{
    width:10px;
    height:10px;
    border-radius:50%;
    flex:0 0 10px;
    box-shadow:0 0 0 5px rgba(93,154,137,.08);
}
.review-link{
    height:1px;
    flex:.45;
    min-width:20px;
    background:linear-gradient(90deg,#D8E4E0,#C7D8D2);
}
.review-summary{
    display:grid;
    grid-template-columns:1.45fr 1fr;
    gap:1rem;
    margin:.15rem 0 .95rem;
    padding:1rem 1.05rem;
    border-radius:18px;
    border:1px solid #DEE7E4;
    background:
        radial-gradient(circle at 92% 14%,rgba(225,218,245,.22),transparent 26%),
        linear-gradient(135deg,#F1F8F5,#FCFDFD 70%);
}
.review-summary-label{
    font-size:.54rem;
    letter-spacing:.15em;
    text-transform:uppercase;
    color:#7A918B;
    font-weight:800;
    margin-bottom:.28rem;
}
.review-summary h3{
    margin:0;
    color:#263A3B;
    font-size:.95rem;
    font-weight:750;
    letter-spacing:-.01em;
}
.review-summary p{
    margin:.36rem 0 0;
    color:#718184;
    font-size:.64rem;
    line-height:1.52;
    max-width:680px;
}
.review-summary-metrics{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:8px;
    align-self:center;
}
.metric-chip{
    padding:.75rem .68rem;
    border-radius:14px;
    background:rgba(255,255,255,.88);
    border:1px solid #E0E8E6;
    text-align:center;
}
.metric-chip:nth-child(1){box-shadow:inset 0 2px 0 #76B29E}
.metric-chip:nth-child(2){box-shadow:inset 0 2px 0 #D6A88E}
.metric-chip:nth-child(3){box-shadow:inset 0 2px 0 #9EAED1}
.metric-chip span{
    display:block;
    color:#334749;
    font-size:.65rem;
    font-weight:800;
}
.metric-chip small{
    display:block;
    margin-top:.12rem;
    color:#8B9799;
    font-size:.52rem;
}
@media(max-width:900px){
    .review-path-track{flex-wrap:wrap}
    .review-link{display:none}
    .review-node{min-width:44%}
    .review-summary{grid-template-columns:1fr}
}
@media(max-width:600px){
    .review-summary-metrics{grid-template-columns:1fr}
    .review-node{min-width:100%}
}


/* ===== FINAL MEDICAL GEN-Z VISUAL PASS ===== */

/* More editorial, less flat */
.stApp{
    background:
        radial-gradient(circle at 8% 4%, rgba(201,230,220,.32), transparent 23%),
        radial-gradient(circle at 91% 8%, rgba(231,224,245,.22), transparent 20%),
        linear-gradient(180deg,#F7FAF9 0%,#F4F7F6 100%) !important;
}

/* Bigger global reading scale */
.section-label{
    font-size:.74rem !important;
    letter-spacing:.28em !important;
}
.section-title{
    font-size:3.45rem !important;
    line-height:.98 !important;
}
.section-copy{
    font-size:1rem !important;
    line-height:1.72 !important;
    max-width:820px !important;
}
.workflow{display:none !important;}

/* Benchmark intro becomes a stronger editorial block */
.section-head{
    position:relative !important;
    overflow:hidden !important;
    padding:1.45rem 1.6rem 1.4rem !important;
    border:1px solid #DCE7E3 !important;
    border-radius:24px !important;
    background:
        linear-gradient(135deg,rgba(255,255,255,.98),rgba(244,250,247,.96)) !important;
    box-shadow:0 18px 40px rgba(37,65,59,.06) !important;
}
.section-head:before{
    content:"";
    position:absolute;
    top:0;
    left:0;
    width:6px;
    height:100%;
    background:linear-gradient(180deg,#5E9D8B,#A4C8BC);
}
.section-head:after{
    display:block !important;
    content:"";
    position:absolute;
    width:170px;
    height:170px;
    border-radius:50%;
    right:-60px;
    top:-75px;
    background:rgba(208,232,224,.34);
}

/* Review path: bigger, cleaner timeline */
.review-path{
    margin:1.15rem 0 1.35rem !important;
    padding:1.2rem 1.25rem 1.25rem !important;
    border-radius:22px !important;
    background:rgba(255,255,255,.92) !important;
    border:1px solid #DCE6E3 !important;
    box-shadow:0 14px 32px rgba(40,68,62,.055) !important;
}
.review-path-title{
    margin-bottom:1.05rem !important;
}
.review-path-title span{
    font-size:.82rem !important;
    letter-spacing:.14em !important;
}
.review-path-title small{
    font-size:.70rem !important;
}
.review-path-track{
    gap:.7rem !important;
}
.review-node{
    padding:.72rem .78rem !important;
    border-radius:16px;
    background:#FAFCFB;
    border:1px solid #E5ECEA;
    transition:transform .18s ease,border-color .18s ease,background .18s ease;
}
.review-node:hover{
    transform:translateY(-2px);
    border-color:#CADDD6;
    background:#FFFFFF;
}
.review-node b{
    font-size:.82rem !important;
}
.review-node small{
    font-size:.66rem !important;
    margin-top:.14rem !important;
}
.review-dot{
    width:12px !important;
    height:12px !important;
    flex-basis:12px !important;
    box-shadow:0 0 0 6px rgba(93,154,137,.08) !important;
}
.review-link{
    min-width:24px !important;
    background:linear-gradient(90deg,#CFDDD9,#BFD2CC) !important;
}

/* Summary: asymmetrical feature panel instead of 3 bland cards */
.review-summary{
    position:relative !important;
    overflow:hidden !important;
    display:grid !important;
    grid-template-columns:1.3fr .95fr !important;
    gap:1.35rem !important;
    padding:1.35rem 1.4rem !important;
    margin:.2rem 0 1.15rem !important;
    border-radius:22px !important;
    border:1px solid #D7E5E0 !important;
    background:
        linear-gradient(135deg,#EAF5F1 0%,#F8FBFA 58%,#F7F3FB 100%) !important;
    box-shadow:0 16px 36px rgba(39,67,61,.055) !important;
}
.review-summary:after{
    content:"";
    position:absolute;
    width:120px;
    height:120px;
    border-radius:50%;
    right:-38px;
    bottom:-55px;
    background:rgba(222,211,242,.28);
}
.review-summary-label{
    font-size:.65rem !important;
    letter-spacing:.16em !important;
}
.review-summary h3{
    font-size:1.25rem !important;
    line-height:1.18 !important;
    margin-top:.1rem !important;
}
.review-summary p{
    font-size:.76rem !important;
    line-height:1.6 !important;
    margin-top:.45rem !important;
    max-width:720px !important;
}
.review-summary-metrics{
    gap:10px !important;
}
.metric-chip{
    min-height:88px !important;
    display:flex;
    flex-direction:column;
    justify-content:center;
    padding:.9rem .75rem !important;
    border-radius:16px !important;
    background:rgba(255,255,255,.90) !important;
    border:1px solid rgba(218,229,225,.95) !important;
    box-shadow:0 8px 18px rgba(45,73,67,.035);
}
.metric-chip span{
    font-size:.78rem !important;
}
.metric-chip small{
    font-size:.60rem !important;
    margin-top:.18rem !important;
}

/* Uploader looks more intentional */
[data-testid="stFileUploader"]{
    border-radius:22px !important;
    padding:1rem !important;
    border:1.5px dashed #AFCFC4 !important;
    background:
        linear-gradient(135deg,#FBFDFC 0%,#F8FBFA 100%) !important;
}
[data-testid="stFileUploader"] section{
    min-height:96px !important;
}
[data-testid="stFileUploader"] p{
    font-size:.95rem !important;
}
[data-testid="stFileUploader"] small{
    font-size:.78rem !important;
}
[data-testid="stFileUploader"] button{
    font-size:.82rem !important;
    padding:.66rem 1rem !important;
}

/* Result section: bigger text and stronger rhythm */
.result-ribbon{
    padding:.9rem 1rem !important;
    border-radius:18px !important;
}
.result-ribbon-text b{
    font-size:.86rem !important;
}
.result-ribbon-text span{
    font-size:.70rem !important;
}
.image-label{
    font-size:.90rem !important;
    padding:.48rem .64rem !important;
}
.model-badge{
    font-size:.76rem !important;
}
.model-status{
    font-size:.63rem !important;
}
.mini-value{
    font-size:1.14rem !important;
}
.mini-label{
    font-size:.62rem !important;
}
.legend-item{
    font-size:.70rem !important;
}
.explain-item{
    padding:1.1rem 1.15rem !important;
}
.explain-item b{
    font-size:.82rem !important;
}
.explain-item span{
    font-size:.74rem !important;
    line-height:1.55 !important;
}

/* Slightly richer medical accents, still restrained */
.info-card:nth-of-type(1){
    background:linear-gradient(180deg,#FFFFFF,#F3F9F6) !important;
}
.info-card:nth-of-type(2){
    background:linear-gradient(180deg,#FFFFFF,#F7F5FB) !important;
}

/* Mobile */
@media(max-width:1000px){
    .review-summary{
        grid-template-columns:1fr !important;
    }
    .review-summary-metrics{
        grid-template-columns:repeat(3,1fr) !important;
    }
}
@media(max-width:700px){
    .section-title{
        font-size:2.55rem !important;
    }
    .section-copy{
        font-size:.9rem !important;
    }
    .review-summary-metrics{
        grid-template-columns:1fr !important;
    }
}


/* ===== WHITE MEDICAL / GEN-Z REDESIGN ===== */
.stApp{
    background:#FFFFFF !important;
}
.block-container{
    max-width:1780px !important;
    width:96% !important;
    padding:0 1.1rem 3.2rem !important;
}

/* airy whitespace + visual separators */
.section-wrap{
    margin-top:1.35rem !important;
}
.review-header{
    position:relative;
    display:grid;
    grid-template-columns:1.5fr .65fr;
    gap:1.2rem;
    align-items:end;
    padding:1.35rem 1.45rem 1.45rem;
    border-radius:24px;
    background:
        radial-gradient(circle at 8% 0%,rgba(205,232,223,.45),transparent 24%),
        linear-gradient(135deg,#F9FCFB 0%,#FFFFFF 70%);
    border:1px solid #DDE7E4;
    box-shadow:0 16px 36px rgba(36,61,56,.055);
    overflow:hidden;
}
.review-header:after{
    content:"";
    position:absolute;
    width:150px;height:150px;border-radius:50%;
    right:-48px;bottom:-88px;
    background:rgba(228,219,246,.34);
}
.review-header-left{position:relative;z-index:2}
.review-header-note{
    position:relative;z-index:2;
    display:flex;align-items:center;gap:.72rem;
    justify-self:end;
    padding:.78rem .92rem;
    border-radius:16px;
    background:#FFFFFF;
    border:1px solid #E1E9E6;
    min-width:220px;
}
.note-dot{
    width:10px;height:10px;border-radius:50%;
    background:#5C9E8B;
    box-shadow:0 0 0 6px rgba(92,158,139,.10);
}
.review-header-note b{
    display:block;color:#314445;font-size:.75rem;
}
.review-header-note small{
    display:block;color:#8A989A;font-size:.62rem;margin-top:.12rem;
}

/* Review rail */
.scan-rail{
    margin:1rem 0 1.25rem;
    display:grid;
    grid-template-columns:.9fr 1.6fr;
    gap:1rem;
    padding:1.15rem;
    border-radius:24px;
    background:#FFFFFF;
    border:1px solid #DEE7E4;
    box-shadow:0 14px 32px rgba(36,61,56,.045);
}
.scan-rail-copy{
    padding:.3rem .4rem;
}
.scan-eyebrow{
    font-size:.60rem;
    letter-spacing:.17em;
    text-transform:uppercase;
    color:#6F9188;
    font-weight:800;
    margin-bottom:.42rem;
}
.scan-title{
    font-family:'Libre Baskerville',Georgia,serif;
    color:#223536;
    font-size:1.55rem;
    line-height:1.02;
    letter-spacing:-.02em;
}
.scan-sub{
    margin-top:.56rem;
    color:#7A898B;
    font-size:.69rem;
    line-height:1.55;
    max-width:340px;
}
.scan-rail-track{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:8px;
    align-items:stretch;
}
.scan-stage{
    position:relative;
    min-height:112px;
    padding:.9rem .8rem;
    border-radius:18px;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
    border:1px solid #E1E9E6;
    background:#FAFCFB;
    overflow:hidden;
}
.scan-stage:after{
    content:"";
    position:absolute;
    width:78px;height:78px;border-radius:50%;
    right:-26px;bottom:-32px;
    opacity:.32;
}
.s1:after{background:#BFDCD4}
.s2:after{background:#DCE7C9}
.s3:after{background:#F2CDBA}
.s4:after{background:#C9DCE9}
.scan-no{
    font-size:.55rem;
    letter-spacing:.14em;
    color:#8B989A;
    font-weight:800;
}
.scan-stage b{
    display:block;
    color:#2C3E40;
    font-size:.76rem;
    position:relative;z-index:2;
}
.scan-stage small{
    display:block;
    color:#899698;
    font-size:.58rem;
    margin-top:.14rem;
    position:relative;z-index:2;
}

/* Case desk */
.case-desk{
    display:grid;
    grid-template-columns:1.35fr .9fr;
    gap:1rem;
    margin:.15rem 0 1rem;
    padding:1.2rem 1.25rem;
    border-radius:24px;
    background:
        linear-gradient(135deg,#F2F9F6 0%,#FFFFFF 58%,#F7F4FB 100%);
    border:1px solid #DDE7E4;
    box-shadow:0 14px 32px rgba(36,61,56,.045);
}
.case-desk-kicker{
    color:#75928A;
    text-transform:uppercase;
    letter-spacing:.15em;
    font-size:.56rem;
    font-weight:800;
    margin-bottom:.35rem;
}
.case-desk-main h3{
    margin:0;
    color:#233638;
    font-size:1.22rem;
    letter-spacing:-.015em;
}
.case-desk-main p{
    margin:.42rem 0 0;
    max-width:690px;
    color:#758487;
    font-size:.72rem;
    line-height:1.58;
}
.case-tags{
    display:flex;
    flex-wrap:wrap;
    gap:.38rem;
    margin-top:.78rem;
}
.case-tags span{
    padding:.38rem .58rem;
    border-radius:999px;
    background:#FFFFFF;
    border:1px solid #DEE7E4;
    color:#6D7D7F;
    font-size:.56rem;
    font-weight:700;
}
.case-desk-side{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:8px;
    align-self:center;
}
.case-metric{
    min-height:88px;
    padding:.75rem .6rem;
    border-radius:16px;
    background:rgba(255,255,255,.92);
    border:1px solid #DEE7E4;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
}
.case-metric:nth-child(1){box-shadow:inset 0 3px 0 #76AF9D}
.case-metric:nth-child(2){box-shadow:inset 0 3px 0 #D4A38A}
.case-metric:nth-child(3){box-shadow:inset 0 3px 0 #9BB3D5}
.case-metric span{
    color:#34484A;
    font-size:.70rem;
    font-weight:800;
}
.case-metric small{
    color:#8B989A;
    font-size:.54rem;
    margin-top:.16rem;
}

/* Uploader as a softer clinical drop-zone */
[data-testid="stFileUploader"]{
    background:#FFFFFF !important;
    border:1.5px dashed #AACBC0 !important;
    border-radius:22px !important;
    padding:1rem !important;
    box-shadow:0 14px 34px rgba(38,63,58,.045) !important;
}
[data-testid="stFileUploader"] section{
    min-height:104px !important;
    background:
        linear-gradient(135deg,#FFFFFF 0%,#F8FBFA 100%) !important;
    border-radius:17px !important;
}
[data-testid="stFileUploader"] p{
    font-size:1rem !important;
    font-weight:600 !important;
}
[data-testid="stFileUploader"] small{
    font-size:.79rem !important;
    color:#77878A !important;
}
[data-testid="stFileUploader"] button{
    background:#245C56 !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:12px !important;
    font-size:.82rem !important;
    font-weight:700 !important;
    padding:.66rem 1.05rem !important;
}

/* keep results clean but richer */
.result-ribbon{
    background:#F5F9F7 !important;
    border:1px solid #DDE7E4 !important;
}
.info-card{
    border-radius:18px !important;
}
.explain-item{
    border-radius:18px !important;
}

@media(max-width:1000px){
    .review-header{grid-template-columns:1fr}
    .review-header-note{justify-self:start}
    .scan-rail{grid-template-columns:1fr}
    .scan-rail-track{grid-template-columns:repeat(2,1fr)}
    .case-desk{grid-template-columns:1fr}
}
@media(max-width:640px){
    .scan-rail-track{grid-template-columns:1fr}
    .case-desk-side{grid-template-columns:1fr}
}


.review-path,.review-summary{display:none !important;}


/* ===== GEN-Z MEDICAL EDITORIAL REDESIGN ===== */
.stApp{
    background:#FFFFFF !important;
}
.block-container{
    max-width:1820px !important;
    width:96% !important;
    padding:0 1.1rem 3.5rem !important;
}

/* subtle paper/grain so the white page doesn't feel dead */
.stApp:before{
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    z-index:0;
    opacity:.10;
    background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitchTiles'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.20'/%3E%3C/svg%3E");
}

/* ticker: editorial break between hero and content */
.medical-ticker{
    position:relative;
    z-index:2;
    margin:1.15rem -1.1rem 1.8rem;
    overflow:hidden;
    background:#173D3A;
    color:#F4FAF8;
    border-top:1px solid #173D3A;
    border-bottom:1px solid #173D3A;
    transform:rotate(-.45deg) scale(1.01);
    box-shadow:0 10px 24px rgba(27,58,54,.08);
}
.medical-ticker-track{
    white-space:nowrap;
    display:inline-block;
    min-width:200%;
    padding:.73rem 0;
    font-size:.62rem;
    font-weight:800;
    letter-spacing:.18em;
    animation:medicalMarquee 28s linear infinite;
}
@keyframes medicalMarquee{
    to{transform:translateX(-50%)}
}

/* main editorial stage */
.med-story{
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:.92fr 1.08fr;
    gap:3.4rem;
    align-items:center;
    min-height:560px;
    margin:0 auto 2rem;
    padding:2.4rem 3.2rem 2.7rem;
    border-radius:34px;
    background:
        radial-gradient(circle at 8% 10%,rgba(205,232,223,.48),transparent 28%),
        radial-gradient(circle at 94% 88%,rgba(228,218,244,.38),transparent 24%),
        linear-gradient(135deg,#F8FCFA 0%,#FFFFFF 56%,#F7F4FB 100%);
    border:1px solid #DDE8E4;
    overflow:hidden;
    box-shadow:0 24px 60px rgba(33,61,55,.06);
}
.med-story:before{
    content:"";
    position:absolute;
    width:240px;height:240px;border-radius:50%;
    border:1px solid rgba(107,161,145,.14);
    left:-105px;top:-75px;
}
.med-story-copy{
    position:relative;
    z-index:3;
    padding-left:.4rem;
}
.med-story-eyebrow{
    display:flex;
    align-items:center;
    gap:.65rem;
    text-transform:uppercase;
    letter-spacing:.22em;
    font-size:.66rem;
    font-weight:800;
    color:#588C7E;
}
.med-story-eyebrow:before{
    content:"";
    width:30px;height:1px;background:#75A697;
}
.med-story h2{
    margin:1.05rem 0 .9rem;
    color:#1D3031;
    font-family:'Libre Baskerville',Georgia,serif;
    font-size:clamp(3.3rem,5.3vw,5.7rem);
    font-weight:400;
    line-height:.94;
    letter-spacing:-.045em;
}
.med-story h2 em{
    display:block;
    color:#4D8D7B;
    font-style:italic;
    font-weight:400;
}
.med-story p{
    max-width:560px;
    margin:0;
    color:#6F8082;
    font-size:.94rem;
    line-height:1.78;
}
.med-story-notes{
    display:flex;
    flex-wrap:wrap;
    gap:.65rem;
    margin-top:1.45rem;
}
.med-note{
    display:flex;
    align-items:center;
    gap:.55rem;
    padding:.58rem .72rem;
    border-radius:15px;
    background:rgba(255,255,255,.86);
    border:1px solid #DEE8E5;
    box-shadow:0 8px 18px rgba(42,67,62,.04);
}
.med-note-no{
    display:grid;
    place-items:center;
    min-width:30px;
    height:30px;
    padding:0 .35rem;
    border-radius:10px;
    background:#E4F1EC;
    color:#3C7667;
    font-size:.58rem;
    font-weight:800;
}
.med-note b{
    display:block;
    color:#304244;
    font-size:.68rem;
    line-height:1.1;
}
.med-note small{
    display:block;
    margin-top:.10rem;
    color:#879496;
    font-size:.55rem;
}

/* visual side */
.med-story-art{
    position:relative;
    min-height:480px;
    display:flex;
    align-items:center;
    justify-content:center;
}
.ct-arch{
    position:relative;
    width:min(90%,480px);
    height:445px;
    border-radius:240px 240px 54px 54px;
    overflow:hidden;
    border:7px solid #FFFFFF;
    background:#DCEAE6;
    box-shadow:
        0 26px 60px rgba(26,56,51,.18),
        inset 0 0 0 1px rgba(255,255,255,.55);
    transform:rotate(2deg);
}
.ct-arch:before{
    content:"";
    position:absolute;
    inset:16px;
    border-radius:220px 220px 43px 43px;
    border:1px solid rgba(255,255,255,.38);
    z-index:3;
    pointer-events:none;
}
.ct-arch img{
    width:100%;
    height:100%;
    object-fit:cover;
    filter:contrast(1.06) brightness(.90);
}
.scan-line{
    position:absolute;
    left:8%;
    right:8%;
    top:18%;
    height:1px;
    background:linear-gradient(90deg,transparent,#A7E0D2,transparent);
    box-shadow:0 0 16px rgba(125,217,195,.65);
    animation:scanMove 4.6s ease-in-out infinite;
}
@keyframes scanMove{
    0%,100%{top:18%;opacity:.25}
    50%{top:80%;opacity:.85}
}

/* floating labels, inspired by sticker-like composition */
.float-label{
    position:absolute;
    z-index:6;
    display:flex;
    align-items:center;
    gap:.45rem;
    padding:.62rem .78rem;
    border-radius:14px;
    background:rgba(255,255,255,.92);
    border:1px solid #DDE7E4;
    color:#435557;
    font-size:.62rem;
    font-weight:800;
    box-shadow:0 14px 30px rgba(34,59,54,.11);
    backdrop-filter:blur(10px);
}
.fl-gt{left:1%;top:19%;transform:rotate(-5deg)}
.fl-y11{right:-1%;top:35%;transform:rotate(5deg)}
.fl-y26{left:6%;bottom:14%;transform:rotate(4deg)}
.float-dot{width:8px;height:8px;border-radius:50%}
.float-dot.gt{background:#62C58A}
.float-dot.y11{background:#F08080}
.float-dot.y26{background:#53BDD0}

/* playful but still clinical accents */
.art-orb{
    position:absolute;
    border-radius:50%;
    z-index:1;
}
.orb-a{
    width:105px;height:105px;
    right:2%;top:8%;
    background:#E7E1F5;
    opacity:.72;
}
.orb-b{
    width:72px;height:72px;
    left:4%;bottom:8%;
    background:#D8EEE7;
    opacity:.86;
}
.art-cross{
    position:absolute;
    right:8%;
    bottom:15%;
    color:#80AE9F;
    font-size:2.1rem;
    font-weight:300;
    transform:rotate(9deg);
}

/* benchmark section: white page but stronger editorial presence */
.review-header{
    margin-top:.5rem !important;
    padding:1.8rem 1.8rem !important;
    border-radius:28px !important;
    background:
        radial-gradient(circle at 96% 5%,rgba(223,215,243,.33),transparent 22%),
        linear-gradient(135deg,#F6FBF8 0%,#FFFFFF 75%) !important;
    border:1px solid #DDE8E4 !important;
    box-shadow:0 18px 42px rgba(34,61,55,.05) !important;
}
.section-label{
    font-size:.72rem !important;
}
.section-title{
    font-size:3.55rem !important;
}
.section-copy{
    font-size:.97rem !important;
    line-height:1.72 !important;
}

/* compact "lab tab" instead of plain box */
.review-header-note{
    border-radius:18px !important;
    background:#173D3A !important;
    color:#FFFFFF !important;
    border:none !important;
    box-shadow:0 10px 24px rgba(23,61,58,.14) !important;
}
.review-header-note b{color:#FFFFFF !important}
.review-header-note small{color:#D7E7E2 !important}
.note-dot{background:#78C5AF !important;box-shadow:0 0 0 6px rgba(120,197,175,.14) !important}

/* uploader feels like a medical workstation drop zone */
[data-testid="stFileUploader"]{
    margin-top:.7rem !important;
    border:1.5px dashed #9FC7BA !important;
    background:
        radial-gradient(circle at 7% 20%,rgba(207,232,224,.30),transparent 18%),
        #FFFFFF !important;
    border-radius:24px !important;
    box-shadow:0 16px 38px rgba(36,62,57,.045) !important;
}
[data-testid="stFileUploader"] section{
    min-height:112px !important;
}
[data-testid="stFileUploader"] button{
    background:#173D3A !important;
    border-radius:999px !important;
    padding:.72rem 1.15rem !important;
}

/* hide obsolete blocks */
.scan-rail,.case-desk,.review-path,.review-summary{display:none !important;}

@media(max-width:1050px){
    .med-story{grid-template-columns:1fr;gap:1.1rem}
    .med-story-copy{text-align:center;padding:0}
    .med-story-eyebrow{justify-content:center}
    .med-story p{margin-left:auto;margin-right:auto}
    .med-story-notes{justify-content:center}
    .med-story-art{min-height:430px}
}
@media(max-width:700px){
    .med-story{padding:1.6rem 1.1rem 2rem;border-radius:24px}
    .med-story h2{font-size:3.2rem}
    .ct-arch{height:360px}
    .float-label{font-size:.55rem}
}


/* ===== PREMIUM CLINICAL EDITORIAL ===== */
.stApp{
    background:#FCFCFB !important;
}
.block-container{
    max-width:1760px !important;
    width:95% !important;
    padding:0 1rem 3.2rem !important;
}
.stApp:before{display:none !important;}

/* hero: quieter, more refined */
.hero{
    box-shadow:0 20px 48px rgba(24,48,44,.10) !important;
    border-radius:0 0 30px 30px !important;
}
.hero-card{
    background:rgba(249,250,249,.94) !important;
    border-radius:24px !important;
    box-shadow:0 22px 58px rgba(8,26,24,.16) !important;
}
.eyebrow{
    background:#E8F0ED !important;
    color:#405F58 !important;
    letter-spacing:.13em !important;
}
.hero-card h1{
    color:#1B292A !important;
    font-size:52px !important;
}
.subtitle{
    color:#657476 !important;
    font-size:14px !important;
}
.tag{
    background:#F4F6F5 !important;
    color:#465758 !important;
    border:1px solid #E6ECEA !important;
}
.t1,.t2,.t3,.t4,.t5{background:#F4F6F5 !important;}

/* clean clinical overview */
.clinical-overview{
    display:grid;
    grid-template-columns:1.05fr .95fr;
    gap:2rem;
    align-items:center;
    margin:2rem 0 1.6rem;
    padding:2rem 2.2rem;
    border-radius:24px;
    background:#FFFFFF;
    border:1px solid #E1E7E5;
    box-shadow:0 14px 36px rgba(29,54,49,.045);
}
.clinical-overview-label{
    font-size:.62rem;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:#6C8A82;
    font-weight:700;
    margin-bottom:.8rem;
}
.clinical-overview h2{
    margin:0;
    font-family:'Libre Baskerville',Georgia,serif;
    color:#1E2F30;
    font-size:clamp(2.8rem,4.4vw,4.7rem);
    line-height:1.02;
    letter-spacing:-.035em;
    font-weight:400;
}
.clinical-overview h2 em{
    color:#567B71;
    font-style:italic;
    font-weight:400;
}
.clinical-overview p{
    margin:1rem 0 0;
    max-width:610px;
    color:#6E7E80;
    font-size:.90rem;
    line-height:1.75;
}
.clinical-overview-panel{
    padding:1.2rem;
    border-radius:20px;
    background:#F7F9F8;
    border:1px solid #E1E8E6;
}
.clinical-overview-panel-head{
    display:flex;
    justify-content:space-between;
    align-items:baseline;
    gap:1rem;
    margin-bottom:1rem;
}
.clinical-overview-panel-head span{
    color:#344749;
    font-size:.70rem;
    font-weight:700;
}
.clinical-overview-panel-head small{
    color:#8A9698;
    font-size:.59rem;
}
.clinical-overview-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px;
}
.clinical-overview-item{
    min-height:100px;
    padding:.9rem;
    border-radius:14px;
    background:#FFFFFF;
    border:1px solid #E2E8E6;
}
.overview-index{
    display:block;
    color:#87A098;
    font-size:.52rem;
    letter-spacing:.12em;
    font-weight:800;
    margin-bottom:.45rem;
}
.clinical-overview-item b{
    display:block;
    color:#314345;
    font-size:.76rem;
    margin-bottom:.18rem;
}
.clinical-overview-item small{
    color:#8A9698;
    font-size:.60rem;
}

/* section header */
.review-header{
    background:#FFFFFF !important;
    border:1px solid #E1E7E5 !important;
    border-radius:22px !important;
    box-shadow:0 12px 30px rgba(27,51,47,.04) !important;
    padding:1.45rem 1.6rem !important;
}
.review-header:after{display:none !important;}
.section-label{
    color:#6F8A83 !important;
    font-size:.64rem !important;
    letter-spacing:.20em !important;
}
.section-title{
    color:#1F3031 !important;
    font-size:3.2rem !important;
}
.section-copy{
    color:#718083 !important;
    font-size:.90rem !important;
}
.review-header-note{
    background:#F6F8F7 !important;
    color:#3E5152 !important;
    border:1px solid #E1E7E5 !important;
    box-shadow:none !important;
}
.review-header-note b{color:#354748 !important;}
.review-header-note small{color:#8A9698 !important;}
.note-dot{
    background:#6F988C !important;
    box-shadow:0 0 0 5px rgba(111,152,140,.10) !important;
}

/* uploader */
[data-testid="stFileUploader"]{
    background:#FFFFFF !important;
    border:1px dashed #B7C9C4 !important;
    border-radius:20px !important;
    box-shadow:none !important;
}
[data-testid="stFileUploader"] section{
    background:#FFFFFF !important;
}
[data-testid="stFileUploader"] button{
    background:#2F5C55 !important;
    color:#FFFFFF !important;
    border-radius:10px !important;
    box-shadow:none !important;
}

/* results */
.result-ribbon{
    background:#F7F9F8 !important;
    border:1px solid #E0E7E5 !important;
    box-shadow:none !important;
}
.benchmark-shell{
    background:#FFFFFF !important;
    border:1px solid #E1E7E5 !important;
    box-shadow:0 12px 28px rgba(26,49,45,.035) !important;
}
.info-card{
    background:#FFFFFF !important;
    border:1px solid #E1E7E5 !important;
    box-shadow:none !important;
}
.info-card:before{
    background:#6D9489 !important;
}
.model-badge{
    background:#E8F0ED !important;
    color:#315E55 !important;
}
.mini-metric{
    background:#F8FAF9 !important;
    border:1px solid #E5EBE9 !important;
}
.mini-value{
    color:#315E55 !important;
}
.explain-item{
    background:#FFFFFF !important;
    border:1px solid #E1E7E5 !important;
    box-shadow:none !important;
}

/* remove previous playful extras */
.medical-ticker,.med-story,.art-orb,.float-label,.scan-line{display:none !important;}

@media(max-width:950px){
    .clinical-overview{grid-template-columns:1fr}
}
@media(max-width:620px){
    .clinical-overview-grid{grid-template-columns:1fr}
}


/* Keep Streamlit error text visible if an exception occurs */
[data-testid="stAlert"]{
    color:#5D2D2D !important;
}
[data-testid="stAlert"] *{
    color:inherit !important;
}


/* ===== MODERN MEDICAL / YOUNG EDITORIAL PASS ===== */
.stApp{
    background:#FFFFFF !important;
}
.block-container{
    max-width:1740px !important;
    width:95% !important;
    padding:0 1.1rem 3.5rem !important;
}

/* Bento section */
.clinical-bento{
    position:relative;
    display:grid;
    grid-template-columns:1.05fr 1fr;
    gap:1.7rem;
    margin:2rem 0 1.7rem;
    padding:2.1rem 2.2rem;
    border-radius:30px;
    background:
        radial-gradient(circle at 8% 8%,rgba(207,232,224,.46),transparent 27%),
        linear-gradient(135deg,#F8FBFA 0%,#FFFFFF 62%,#F7F5FB 100%);
    border:1px solid #DDE7E4;
    overflow:hidden;
    box-shadow:0 18px 42px rgba(27,50,46,.05);
}
.clinical-bento:after{
    content:"";
    position:absolute;
    width:220px;height:220px;border-radius:50%;
    right:-95px;bottom:-105px;
    border:1px solid rgba(83,132,119,.12);
}
.clinical-bento-copy{
    position:relative;
    z-index:2;
    padding:.35rem .35rem .25rem;
}
.bento-kicker{
    display:inline-flex;
    align-items:center;
    gap:.6rem;
    font-size:.61rem;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:#5D8479;
    font-weight:800;
}
.bento-kicker:before{
    content:"";
    width:28px;height:1px;background:#709A8E;
}
.clinical-bento h2{
    margin:1rem 0 .9rem;
    font-family:'Libre Baskerville',Georgia,serif;
    color:#1D2E2F;
    font-size:clamp(3rem,4.8vw,5rem);
    line-height:.98;
    letter-spacing:-.04em;
    font-weight:400;
}
.clinical-bento h2 em{
    display:block;
    color:#567C71;
    font-style:italic;
    font-weight:400;
}
.clinical-bento p{
    margin:0;
    max-width:590px;
    color:#6F7F81;
    font-size:.9rem;
    line-height:1.75;
}
.bento-rule{
    display:flex;
    flex-wrap:wrap;
    gap:.42rem;
    margin-top:1.2rem;
}
.bento-rule span{
    padding:.38rem .58rem;
    border-radius:999px;
    border:1px solid #DCE6E3;
    background:rgba(255,255,255,.86);
    color:#647779;
    font-size:.56rem;
    font-weight:750;
    letter-spacing:.03em;
}

.clinical-bento-grid{
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:1fr 1fr;
    grid-template-rows:1.15fr .85fr;
    gap:10px;
    min-height:330px;
}
.bento-card{
    position:relative;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
    padding:1rem 1.05rem;
    border-radius:20px;
    overflow:hidden;
    border:1px solid rgba(219,229,226,.95);
}
.bento-card:after{
    content:"";
    position:absolute;
    width:90px;height:90px;border-radius:50%;
    right:-28px;top:-30px;
    background:rgba(255,255,255,.24);
}
.bento-dark{
    background:#173D3A;
    color:#FFFFFF;
    grid-row:1 / span 2;
}
.bento-mint{background:#E9F4F0;}
.bento-ivory{background:#FBF8F3;}
.bento-blue{background:#EEF3F7;}
.bento-no{
    font-size:.58rem;
    font-weight:800;
    letter-spacing:.15em;
    opacity:.72;
}
.bento-card b{
    display:block;
    font-size:.88rem;
    letter-spacing:-.01em;
}
.bento-card small{
    display:block;
    margin-top:.14rem;
    font-size:.60rem;
    opacity:.68;
}
.bento-dark b{font-size:1rem;}
.bento-dark small{color:#D5E5E0;opacity:1}

/* Workspace header */
.workspace-head{
    display:grid;
    grid-template-columns:auto 1fr auto;
    gap:1.2rem;
    align-items:center;
    margin-top:.6rem;
    padding:1.35rem 1.5rem;
    border-radius:24px;
    background:#FFFFFF;
    border:1px solid #DEE7E4;
    box-shadow:0 12px 30px rgba(28,52,47,.04);
}
.workspace-index{
    width:54px;height:54px;
    display:grid;
    place-items:center;
    border-radius:16px;
    background:#173D3A;
    color:#FFFFFF;
    font-size:.74rem;
    font-weight:800;
    letter-spacing:.08em;
}
.workspace-copy .section-label{
    font-size:.60rem !important;
    color:#6E8B83 !important;
    letter-spacing:.18em !important;
}
.workspace-copy .section-title{
    margin-top:.22rem !important;
    font-size:2.75rem !important;
    line-height:1 !important;
}
.workspace-copy .section-copy{
    margin-top:.45rem !important;
    font-size:.82rem !important;
    max-width:760px !important;
}
.workspace-meta{
    min-width:165px;
    padding-left:1rem;
    border-left:1px solid #E1E8E5;
}
.workspace-meta-line{
    display:flex;
    align-items:center;
    gap:.48rem;
    color:#758587;
    font-size:.60rem;
    font-weight:650;
    margin:.35rem 0;
}
.workspace-meta-line span{
    width:6px;height:6px;border-radius:50%;
    background:#6B9D90;
}

/* Uploader */
[data-testid="stFileUploader"]{
    margin-top:.85rem !important;
    background:
        radial-gradient(circle at 7% 20%,rgba(207,232,224,.26),transparent 20%),
        #FFFFFF !important;
    border:1.5px dashed #AFCBC2 !important;
    border-radius:24px !important;
    box-shadow:0 14px 34px rgba(36,63,58,.04) !important;
}
[data-testid="stFileUploader"] section{
    min-height:106px !important;
}
[data-testid="stFileUploader"] button{
    background:#173D3A !important;
    border-radius:999px !important;
    padding:.68rem 1.05rem !important;
}

/* Results get slightly more energy but stay clinical */
.result-ribbon{
    border-radius:18px !important;
    background:linear-gradient(90deg,#F4F8F6,#FAFBFA) !important;
}
.image-label{
    border-radius:999px !important;
}
.info-card{
    border-radius:18px !important;
}
.explain-item{
    border-radius:18px !important;
}

/* eliminate old layouts */
.clinical-overview,.review-header,.review-path,.review-summary,.scan-rail,.case-desk{display:none !important;}

@media(max-width:980px){
    .clinical-bento{grid-template-columns:1fr}
    .clinical-bento-grid{min-height:300px}
    .workspace-head{grid-template-columns:auto 1fr}
    .workspace-meta{grid-column:2;border-left:0;padding-left:0;display:flex;gap:.8rem;flex-wrap:wrap}
}
@media(max-width:620px){
    .clinical-bento{padding:1.45rem 1.15rem;border-radius:24px}
    .clinical-bento-grid{grid-template-columns:1fr;grid-template-rows:auto}
    .bento-dark{grid-row:auto}
    .workspace-head{grid-template-columns:1fr}
    .workspace-index{width:46px;height:46px}
    .workspace-meta{grid-column:auto}
}


/* ===== CLEAN MODERN MEDICAL EDITORIAL ===== */
.stApp{
    background:#FFFFFF !important;
}
.block-container{
    max-width:1700px !important;
    width:95% !important;
    padding:0 1.1rem 3.6rem !important;
}

/* remove old blocks if any remain */
.clinical-bento,
.workspace-head,
.review-header-note,
.review-header,
.clinical-overview,
.review-path,
.review-summary,
.scan-rail,
.case-desk{
    display:none !important;
}

/* one strong workspace section */
.workspace-stage{
    position:relative;
    margin:2.1rem 0 1rem;
    padding:1.8rem 2rem 1.45rem;
    border-radius:28px;
    background:
        linear-gradient(135deg,#F8FBFA 0%,#FFFFFF 68%);
    border:1px solid #DDE7E4;
    box-shadow:0 18px 44px rgba(29,54,49,.055);
    overflow:hidden;
}
.workspace-stage:before{
    content:"01";
    position:absolute;
    right:1.3rem;
    top:-1.25rem;
    font-family:'Libre Baskerville',Georgia,serif;
    font-size:8.5rem;
    line-height:1;
    color:rgba(39,83,73,.045);
    pointer-events:none;
}
.workspace-stage:after{
    content:"";
    position:absolute;
    left:0;
    top:0;
    width:5px;
    height:100%;
    background:#315F56;
}
.workspace-stage-top{
    position:relative;
    z-index:2;
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:1.2rem;
}
.workspace-kicker{
    color:#6D8F86;
    font-size:.64rem;
    font-weight:800;
    letter-spacing:.19em;
    text-transform:uppercase;
    margin-bottom:.45rem;
}
.workspace-title{
    font-family:'Libre Baskerville',Georgia,serif;
    color:#1E3031;
    font-size:3.25rem;
    line-height:.98;
    letter-spacing:-.035em;
}
.workspace-sub{
    max-width:780px;
    margin-top:.65rem;
    color:#6F8082;
    font-size:.9rem;
    line-height:1.68;
}
.workspace-chip{
    position:relative;
    z-index:2;
    flex:0 0 auto;
    margin-top:.25rem;
    padding:.58rem .78rem;
    border-radius:999px;
    background:#173D3A;
    color:#F5FAF8;
    font-size:.57rem;
    font-weight:800;
    letter-spacing:.12em;
    box-shadow:0 8px 20px rgba(23,61,58,.12);
}
.workspace-track{
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;
    gap:.65rem;
    align-items:center;
    margin-top:1.35rem;
    padding-top:1.1rem;
    border-top:1px solid #E2EAE7;
}
.workspace-step{
    display:flex;
    align-items:center;
    gap:.65rem;
    min-width:0;
    padding:.65rem .72rem;
    border-radius:14px;
    transition:background .18s ease,transform .18s ease;
}
.workspace-step:hover{
    background:#F3F7F5;
    transform:translateY(-1px);
}
.workspace-step > span{
    display:grid;
    place-items:center;
    width:31px;
    height:31px;
    flex:0 0 31px;
    border-radius:10px;
    background:#E4F0EC;
    color:#35695D;
    font-size:.58rem;
    font-weight:800;
}
.workspace-step:nth-of-type(3) > span{
    background:#F7EBE4;
    color:#8B5B46;
}
.workspace-step:nth-of-type(4) > span{
    background:#EAF0F6;
    color:#526D86;
}
.workspace-step b{
    display:block;
    color:#324547;
    font-size:.73rem;
    line-height:1.18;
}
.workspace-step small{
    display:block;
    margin-top:.12rem;
    color:#8A9799;
    font-size:.58rem;
}
.workspace-arrow{
    color:#A5B2B0;
    font-size:.95rem;
    font-weight:600;
}

/* uploader: cleaner + larger */
[data-testid="stFileUploader"]{
    margin-top:.72rem !important;
    background:#FFFFFF !important;
    border:1.5px dashed #AFCBC1 !important;
    border-radius:22px !important;
    box-shadow:0 12px 32px rgba(33,58,53,.04) !important;
    padding:.95rem !important;
}
[data-testid="stFileUploader"] section{
    min-height:108px !important;
    background:
        linear-gradient(135deg,#FFFFFF 0%,#F8FBFA 100%) !important;
    border-radius:16px !important;
}
[data-testid="stFileUploader"] p{
    font-size:.98rem !important;
    font-weight:600 !important;
}
[data-testid="stFileUploader"] small{
    font-size:.76rem !important;
    color:#768689 !important;
}
[data-testid="stFileUploader"] button{
    background:#173D3A !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:999px !important;
    padding:.7rem 1.05rem !important;
    font-size:.8rem !important;
    font-weight:700 !important;
    box-shadow:none !important;
}

/* hero: slightly cleaner, less crowded */
.hero-micro{
    opacity:.72 !important;
}
.ct-layer{
    width:30% !important;
    right:4.2% !important;
    top:10% !important;
    height:80% !important;
}
.hero-card{
    width:min(700px,65%) !important;
}
.hero-card h1{
    font-size:50px !important;
}

/* keep result area refined */
.result-ribbon{
    border-radius:18px !important;
}
.benchmark-shell{
    border-radius:20px !important;
}
.info-card{
    border-radius:18px !important;
}

@media(max-width:980px){
    .workspace-stage-top{flex-direction:column}
    .workspace-track{
        grid-template-columns:1fr 1fr;
    }
    .workspace-arrow{display:none}
    .workspace-chip{margin-top:0}
    .ct-layer{display:none !important}
    .hero-card{left:50% !important;width:85% !important}
}
@media(max-width:620px){
    .workspace-title{font-size:2.45rem}
    .workspace-track{grid-template-columns:1fr}
    .workspace-stage{padding:1.35rem 1.15rem}
}


/* ===== FINAL ELEGANT CLINICAL DESIGN ===== */
:root{
    --paper:#FCFCFA;
    --white:#FFFFFF;
    --ink:#1E2C2E;
    --muted:#728184;
    --teal:#274F49;
    --teal-2:#5F8B80;
    --sage:#E7EFEC;
    --line:#DDE5E2;
    --warm:#F6F1EB;
    --blue:#EDF2F5;
}

.stApp{
    background:
        linear-gradient(180deg,#FCFCFA 0%,#F8FAF9 100%) !important;
}
.block-container{
    max-width:1680px !important;
    width:94% !important;
    padding:0 1.25rem 3.8rem !important;
}

/* ---------- Hero ---------- */
.hero{
    height:590px !important;
    border-radius:0 0 28px 28px !important;
    border:1px solid #DCE4E1 !important;
    box-shadow:0 22px 54px rgba(22,45,41,.10) !important;
}
.hero-card{
    left:42% !important;
    width:min(680px,61%) !important;
    padding:42px 44px 38px !important;
    border-radius:24px !important;
    background:rgba(249,250,249,.94) !important;
    border:1px solid rgba(255,255,255,.88) !important;
    box-shadow:0 24px 62px rgba(6,22,20,.16) !important;
}
.hero-card h1{
    font-size:50px !important;
    line-height:.98 !important;
    letter-spacing:-.035em !important;
    color:var(--ink) !important;
}
.eyebrow{
    background:#E8F0ED !important;
    color:#365E55 !important;
    padding:8px 13px !important;
    font-size:10px !important;
    letter-spacing:.16em !important;
}
.subtitle{
    max-width:590px !important;
    color:#647476 !important;
    font-size:13.5px !important;
    line-height:1.72 !important;
}
.tags{
    margin-top:22px !important;
    gap:7px !important;
}
.tag{
    background:#F6F8F7 !important;
    border:1px solid #E3E9E7 !important;
    color:#485A5C !important;
    padding:7px 11px !important;
    font-size:10px !important;
}
.t1,.t2,.t3,.t4,.t5{background:#F6F8F7 !important;}

.ct-layer{
    width:29% !important;
    right:4.5% !important;
    top:10% !important;
    height:80% !important;
    border-radius:22px !important;
    opacity:.82 !important;
    box-shadow:0 22px 48px rgba(0,0,0,.20) !important;
}
.ct-layer:after{
    left:14px !important;
    top:12px !important;
    padding:5px 8px !important;
    font-size:8px !important;
    letter-spacing:.14em !important;
}
.i3l-mark{
    width:60px !important;
    left:24px !important;
    top:22px !important;
}
.gpt-credit{
    right:24px !important;
    top:24px !important;
    font-size:8.5px !important;
    color:rgba(240,246,244,.85) !important;
}
.gpt-credit img{
    width:22px !important;
    height:22px !important;
}
.hero-micro{
    left:24px !important;
    bottom:20px !important;
    font-size:8.5px !important;
    letter-spacing:.10em !important;
}

/* ---------- Main workspace ---------- */
.workspace-stage{
    margin:2.2rem 0 1.1rem !important;
    padding:2.15rem 2.2rem 1.55rem !important;
    border-radius:26px !important;
    background:
        linear-gradient(135deg,#FFFFFF 0%,#FBFCFB 72%) !important;
    border:1px solid var(--line) !important;
    box-shadow:0 16px 38px rgba(27,50,46,.045) !important;
}
.workspace-stage:before{
    content:"";
    position:absolute;
    width:180px;
    height:180px;
    border-radius:50%;
    right:-70px;
    top:-85px;
    background:rgba(223,233,229,.34);
}
.workspace-stage:after{
    width:4px !important;
    background:var(--teal) !important;
}
.workspace-stage-top{
    align-items:flex-end !important;
}
.workspace-kicker{
    color:#6A877F !important;
    font-size:.61rem !important;
    letter-spacing:.21em !important;
    margin-bottom:.5rem !important;
}
.workspace-title{
    color:var(--ink) !important;
    font-size:3.45rem !important;
    letter-spacing:-.035em !important;
}
.workspace-sub{
    max-width:820px !important;
    margin-top:.7rem !important;
    color:#6E7D80 !important;
    font-size:.91rem !important;
    line-height:1.72 !important;
}
.workspace-chip{
    background:#EEF3F1 !important;
    color:#49675F !important;
    border:1px solid #DDE6E3 !important;
    box-shadow:none !important;
    font-size:.55rem !important;
    letter-spacing:.14em !important;
}

/* Elegant workflow rail */
.workspace-track{
    margin-top:1.55rem !important;
    padding-top:1.25rem !important;
    gap:.8rem !important;
    border-top:1px solid #E2E8E6 !important;
}
.workspace-step{
    min-height:74px !important;
    padding:.72rem .78rem !important;
    border-radius:14px !important;
    background:#FBFCFB;
    border:1px solid #E7ECEA;
}
.workspace-step:hover{
    transform:none !important;
    background:#F8FAF9 !important;
}
.workspace-step > span{
    width:32px !important;
    height:32px !important;
    flex-basis:32px !important;
    background:#E7F0ED !important;
    color:#3D6A60 !important;
}
.workspace-step:nth-of-type(3) > span{
    background:#F4ECE7 !important;
    color:#815E4C !important;
}
.workspace-step:nth-of-type(4) > span{
    background:#EBF0F4 !important;
    color:#526B7C !important;
}
.workspace-step b{
    font-size:.74rem !important;
}
.workspace-step small{
    font-size:.58rem !important;
}
.workspace-arrow{
    color:#A7B3B1 !important;
    font-size:.9rem !important;
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"]{
    margin-top:.8rem !important;
    background:#FFFFFF !important;
    border:1px dashed #B5C8C2 !important;
    border-radius:20px !important;
    padding:.9rem !important;
    box-shadow:0 10px 26px rgba(29,52,47,.035) !important;
}
[data-testid="stFileUploader"] section{
    min-height:102px !important;
    background:linear-gradient(135deg,#FFFFFF 0%,#FAFBFA 100%) !important;
    border-radius:14px !important;
}
[data-testid="stFileUploader"] p{
    font-size:.96rem !important;
    font-weight:600 !important;
}
[data-testid="stFileUploader"] small{
    font-size:.75rem !important;
    color:#7A898C !important;
}
[data-testid="stFileUploader"] button{
    background:var(--teal) !important;
    border-radius:999px !important;
    padding:.68rem 1.05rem !important;
    font-size:.78rem !important;
    font-weight:700 !important;
}

/* ---------- Results ---------- */
.result-ribbon{
    margin-top:1.2rem !important;
    border-radius:16px !important;
    background:#F7F9F8 !important;
    border:1px solid #E0E7E5 !important;
    box-shadow:none !important;
}
.result-ribbon-icon{
    background:#E4EEEA !important;
    color:#355F56 !important;
}
.result-ribbon-text b{
    font-size:.82rem !important;
}
.result-ribbon-text span{
    font-size:.67rem !important;
}
.benchmark-shell{
    background:#FFFFFF !important;
    border:1px solid #E1E7E5 !important;
    border-radius:20px !important;
    box-shadow:0 12px 28px rgba(26,49,45,.035) !important;
}
.image-label{
    background:#FFFFFF !important;
    border:1px solid #E2E8E6 !important;
    box-shadow:none !important;
    font-size:.78rem !important;
}
[data-testid="stImage"]{
    border-radius:16px !important;
    box-shadow:0 8px 20px rgba(31,55,50,.035) !important;
}
.info-card{
    border-radius:16px !important;
    box-shadow:none !important;
    background:#FFFFFF !important;
}
.info-card:before{
    background:#6F968B !important;
}
.model-badge{
    background:#E9F1EE !important;
    color:#315D54 !important;
}
.mini-metric{
    background:#F8FAF9 !important;
    border:1px solid #E5EBE9 !important;
}
.mini-value{
    color:#315D54 !important;
}
.legend-card{
    border-radius:14px !important;
    box-shadow:none !important;
}
.explain-strip{
    gap:10px !important;
}
.explain-item{
    background:#FFFFFF !important;
    border:1px solid #E1E7E5 !important;
    box-shadow:none !important;
    border-radius:16px !important;
}

/* ---------- Responsive ---------- */
@media(max-width:980px){
    .hero-card{
        left:50% !important;
        width:84% !important;
    }
    .ct-layer{display:none !important;}
    .workspace-stage-top{flex-direction:column;align-items:flex-start !important;}
    .workspace-track{
        grid-template-columns:1fr 1fr !important;
    }
    .workspace-arrow{display:none !important;}
}
@media(max-width:620px){
    .workspace-title{
        font-size:2.55rem !important;
    }
    .workspace-track{
        grid-template-columns:1fr !important;
    }
}



/* ===== PREMIUM MEDICAL MOCKUP DIRECTION ===== */
.stApp{
    background:#FFFFFF !important;
}
.block-container{
    max-width:1700px !important;
    width:95% !important;
    padding:0 1.1rem 3.6rem !important;
}

/* remove old workspace variants */
.workspace-stage,.clinical-bento,.workspace-head,.review-header,
.clinical-overview,.review-path,.review-summary,.scan-rail,.case-desk{
    display:none !important;
}

/* case workspace */
.case-workspace{
    position:relative;
    margin:2rem 0 1rem;
    padding:1.9rem 2rem 1.8rem;
    border-radius:28px;
    background:
        radial-gradient(circle at 94% 14%, rgba(229,241,237,.95), transparent 16%),
        linear-gradient(135deg,#FFFFFF 0%,#FBFCFB 100%);
    border:1px solid #DBE6E2;
    box-shadow:0 18px 42px rgba(28,51,46,.05);
    overflow:hidden;
}
.case-copy{
    position:relative;
    z-index:2;
    max-width:860px;
}
.case-kicker{
    color:#6B8E84;
    font-size:.66rem;
    font-weight:800;
    letter-spacing:.20em;
    text-transform:uppercase;
    margin-bottom:.42rem;
}
.case-title{
    font-family:'Libre Baskerville',Georgia,serif;
    color:#1D2F30;
    font-size:3.4rem;
    line-height:.98;
    letter-spacing:-.035em;
}
.case-sub{
    max-width:920px;
    margin-top:.68rem;
    color:#6F7F81;
    font-size:.92rem;
    line-height:1.72;
}
.case-lung-art{
    position:absolute;
    z-index:1;
    right:2rem;
    top:1rem;
    width:155px;
    height:155px;
    opacity:.75;
    pointer-events:none;
}
.case-lung-art img{
    width:100%;
    height:100%;
    object-fit:contain;
}

/* step flow */
.case-flow{
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;
    gap:.72rem;
    align-items:center;
    margin-top:1.45rem;
}
.case-step{
    min-height:166px;
    padding:.82rem;
    border-radius:20px;
    border:1px solid #DFE8E5;
    box-shadow:0 10px 22px rgba(30,54,49,.03);
}
.step-1{background:linear-gradient(180deg,#F6FBF8 0%,#F1F7F4 100%)}
.step-2{background:linear-gradient(180deg,#FBF8F4 0%,#F7F2EC 100%)}
.step-3{background:linear-gradient(180deg,#F3FAF7 0%,#ECF7F3 100%)}
.step-4{background:linear-gradient(180deg,#F3F7FA 0%,#EDF3F7 100%)}

.step-thumb{
    position:relative;
    width:100%;
    height:82px;
    overflow:hidden;
    border-radius:14px;
    border:1px solid rgba(212,223,219,.9);
    background:#0E1617;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.04);
}
.step-thumb img{
    width:100%;
    height:100%;
    object-fit:cover;
    opacity:.94;
    filter:contrast(1.05) brightness(.92);
}
.thumb-mask{
    position:absolute;
    right:20%;
    top:16%;
    width:23%;
    height:58%;
    border-radius:45% 50% 42% 46%;
    opacity:.88;
    filter:drop-shadow(0 0 8px rgba(255,255,255,.08));
}
.mask-green{background:rgba(88,201,126,.72)}
.mask-coral{background:rgba(233,102,102,.76)}
.mask-sky{background:rgba(85,203,220,.78)}

.step-info{
    display:flex;
    align-items:flex-start;
    gap:.7rem;
    margin-top:.78rem;
}
.step-number{
    width:36px;
    height:36px;
    flex:0 0 36px;
    display:grid;
    place-items:center;
    border-radius:12px;
    background:#FFFFFF;
    border:1px solid rgba(214,224,221,.95);
    color:#315F56;
    font-size:.63rem;
    font-weight:800;
    box-shadow:0 5px 12px rgba(34,58,53,.04);
}
.step-2 .step-number{color:#7A6B4E}
.step-3 .step-number{color:#417A6D}
.step-4 .step-number{color:#4E6F86}

.step-body b{
    display:block;
    color:#314345;
    font-size:.8rem;
    line-height:1.18;
}
.step-body small{
    display:block;
    margin-top:.28rem;
    color:#7D8D8F;
    font-size:.64rem;
    line-height:1.45;
}

.case-arrow{
    width:30px;
    height:30px;
    display:grid;
    place-items:center;
    border-radius:50%;
    background:#FFFFFF;
    border:1px solid #DCE6E3;
    color:#6F8D84;
    font-size:1rem;
    box-shadow:0 5px 12px rgba(35,59,54,.035);
}

/* uploader centered */
[data-testid="stFileUploader"]{
    margin-top:.95rem !important;
    background:#FFFFFF !important;
    border:1.5px dashed #AFC8C0 !important;
    border-radius:24px !important;
    padding:1rem !important;
    box-shadow:0 14px 34px rgba(30,54,49,.035) !important;
}
[data-testid="stFileUploader"] section{
    min-height:122px !important;
    background:linear-gradient(135deg,#FFFFFF 0%,#FAFBFA 100%) !important;
    border-radius:17px !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
}
[data-testid="stFileUploader"] section > div{
    width:100% !important;
    text-align:center !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{
    text-align:center !important;
}
[data-testid="stFileUploader"] p{
    font-size:1rem !important;
    font-weight:600 !important;
    text-align:center !important;
    margin-left:auto !important;
    margin-right:auto !important;
}
[data-testid="stFileUploader"] small{
    font-size:.76rem !important;
    color:#7A888B !important;
}
[data-testid="stFileUploader"] button{
    background:#274F49 !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:12px !important;
    padding:.72rem 1.08rem !important;
    font-size:.80rem !important;
    font-weight:700 !important;
    box-shadow:none !important;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]{
    margin-left:auto !important;
    margin-right:auto !important;
}

/* results */
.result-ribbon{
    border-radius:18px !important;
    background:#F6F9F7 !important;
    border:1px solid #E0E7E5 !important;
}
.benchmark-shell{
    border-radius:20px !important;
    background:#FFFFFF !important;
    border:1px solid #E1E7E5 !important;
}
.info-card{border-radius:17px !important;}
.explain-item{border-radius:17px !important;}

@media(max-width:1000px){
    .case-flow{
        grid-template-columns:1fr 1fr;
    }
    .case-arrow{
        display:none;
    }
    .case-lung-art{
        opacity:.45;
        right:1rem;
    }
}
@media(max-width:640px){
    .case-workspace{
        padding:1.35rem 1.15rem;
    }
    .case-title{
        font-size:2.45rem;
    }
    .case-flow{
        grid-template-columns:1fr;
    }
    .case-lung-art{
        display:none;
    }
}

/* ===== FINAL TRUE-DATA PREVIEW + CENTERED UPLOAD ===== */
.case-workspace-intro{
    min-height:210px !important;
    display:flex;
    align-items:center;
    padding:2rem 2.1rem !important;
}
.generated-lungs{
    right:2.5rem !important;
    top:50% !important;
    transform:translateY(-50%);
    width:245px !important;
    height:190px !important;
    opacity:.72 !important;
}
.generated-lungs img{
    object-fit:contain !important;
    filter:saturate(.78) opacity(.92);
}

.upload-intro{
    text-align:center;
    margin:.8rem 0 -.15rem;
}
.upload-icon{
    width:48px;
    height:48px;
    margin:0 auto .55rem;
    display:grid;
    place-items:center;
    border-radius:50%;
    background:#E9F3EF;
    color:#315E55;
    font-size:1.35rem;
    font-weight:700;
    border:1px solid #D8E8E2;
}
.upload-title{
    font-family:'Libre Baskerville',Georgia,serif;
    color:#223536;
    font-size:1.55rem;
}
.upload-sub{
    margin-top:.28rem;
    color:#7B8A8C;
    font-size:.78rem;
}

/* center Streamlit uploader content as much as Streamlit DOM permits */
[data-testid="stFileUploader"]{
    max-width:100% !important;
    margin:0 auto 1rem !important;
}
[data-testid="stFileUploader"] section{
    min-height:140px !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    text-align:center !important;
}
[data-testid="stFileUploader"] section > div{
    width:100% !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
    justify-content:center !important;
    text-align:center !important;
}
[data-testid="stFileUploader"] button{
    margin-left:auto !important;
    margin-right:auto !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small{
    text-align:center !important;
}

/* Actual, data-driven 4-view preview */
.actual-review{
    margin:1.3rem 0 1.1rem;
    padding:1.25rem 1.3rem 1.35rem;
    border-radius:24px;
    background:linear-gradient(135deg,#FFFFFF 0%,#F9FBFA 100%);
    border:1px solid #DDE7E4;
    box-shadow:0 14px 34px rgba(30,54,49,.04);
}
.actual-review-head{
    display:flex;
    justify-content:space-between;
    align-items:end;
    gap:1rem;
    margin-bottom:1rem;
}
.actual-kicker{
    color:#6D8E85;
    font-size:.60rem;
    letter-spacing:.18em;
    text-transform:uppercase;
    font-weight:800;
}
.actual-title{
    margin-top:.28rem;
    font-family:'Libre Baskerville',Georgia,serif;
    color:#203334;
    font-size:2rem;
    line-height:1;
}
.actual-note{
    color:#849294;
    font-size:.62rem;
}
.actual-flow{
    display:grid;
    grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;
    gap:.7rem;
    align-items:center;
}
.actual-card{
    padding:.72rem;
    border-radius:18px;
    border:1px solid #DFE7E4;
    background:#F7FAF8;
    display:flex;
    align-items:center;
    gap:.72rem;
    min-height:102px;
}
.actual-gt{background:#FBF8F3;}
.actual-y11{background:#F2F9F6;}
.actual-y26{background:#F1F5F8;}
.actual-thumb{
    width:82px;
    height:82px;
    flex:0 0 82px;
    overflow:hidden;
    border-radius:13px;
    background:#11191A;
    border:1px solid #DCE5E2;
}
.actual-thumb img{
    width:100%;
    height:100%;
    object-fit:contain;
    object-position:center;
    display:block;
    background:#0F1516;
}
.actual-meta{
    display:flex;
    align-items:flex-start;
    gap:.58rem;
    margin-top:0;
    flex:1;
}
.actual-num{
    width:30px;
    height:30px;
    flex:0 0 30px;
    display:grid;
    place-items:center;
    border-radius:10px;
    background:#FFFFFF;
    border:1px solid #DCE5E2;
    color:#35645A;
    font-size:.58rem;
    font-weight:800;
}
.actual-meta b{
    display:block;
    color:#304345;
    font-size:.73rem;
}
.actual-meta small{
    display:block;
    margin-top:.14rem;
    color:#839194;
    font-size:.56rem;
    line-height:1.35;
}
.actual-arrow{
    width:28px;
    height:28px;
    display:grid;
    place-items:center;
    border-radius:50%;
    background:#FFFFFF;
    border:1px solid #DCE5E2;
    color:#719087;
    font-size:.95rem;
}
@media(max-width:1000px){
    .generated-lungs{width:180px !important;opacity:.35 !important;}
    .actual-flow{grid-template-columns:1fr 1fr;}
    .actual-arrow{display:none;}
}
@media(max-width:640px){
    .generated-lungs{display:none;}
    .actual-flow{grid-template-columns:1fr;}
    .actual-review-head{align-items:flex-start;flex-direction:column;}
}


/* ===== FINAL APPROVED MOCKUP MATCH ===== */
.stApp{
    background:#FFFFFF !important;
}
.block-container{
    max-width:1700px !important;
    width:95% !important;
    padding:0 1.15rem 3.6rem !important;
}

/* hide older pre-upload variants */
.case-workspace-intro,
.case-workspace,
.case-lung-art{
    display:none !important;
}

/* workspace */
.mock-workspace{
    position:relative;
    margin:2rem 0 1rem;
    padding:1.8rem 2rem 1.6rem;
    border-radius:26px;
    overflow:hidden;
    background:
        radial-gradient(circle at 94% 12%, rgba(220,236,231,.76), transparent 18%),
        linear-gradient(135deg,#FFFFFF 0%,#FBFCFB 100%);
    border:1px solid #DBE7E3;
    box-shadow:0 16px 40px rgba(27,51,46,.045);
}
.mock-workspace-head{
    position:relative;
    z-index:2;
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:1rem;
}
.mock-workspace-copy{
    max-width:900px;
}
.mock-kicker{
    color:#6B8F84;
    font-size:.64rem;
    font-weight:800;
    letter-spacing:.21em;
    text-transform:uppercase;
    margin-bottom:.45rem;
}
.mock-title{
    font-family:'Libre Baskerville',Georgia,serif;
    color:#1D3031;
    font-size:3.4rem;
    line-height:.98;
    letter-spacing:-.035em;
}
.mock-sub{
    max-width:920px;
    margin-top:.68rem;
    color:#6F7F81;
    font-size:.92rem;
    line-height:1.72;
}

/* actual generated lung artwork, kept subtle */
.mock-lung{
    position:absolute;
    z-index:1;
    right:1.15rem;
    top:-1rem;
    width:260px;
    height:178px;
    opacity:.72;
    pointer-events:none;
}
.mock-lung img{
    width:100%;
    height:100%;
    object-fit:contain;
    object-position:center top;
    mix-blend-mode:multiply;
}

/* 4-step rail */
.mock-flow{
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;
    gap:.65rem;
    align-items:center;
    margin-top:1.45rem;
}
.mock-step{
    min-height:118px;
    border-radius:19px;
    padding:.72rem;
    border:1px solid #DCE7E3;
    box-shadow:0 8px 18px rgba(31,55,50,.025);
    display:flex;
    align-items:center;
    gap:.72rem;
}
.step-a{background:#F2F8F5}
.step-b{background:#F8F5EF}
.step-c{background:#F0F8F6}
.step-d{background:#F1F5F8}
.mock-thumb{
    width:88px;
    height:88px;
    flex:0 0 88px;
    overflow:hidden;
    border-radius:14px;
    border:1px solid rgba(213,224,220,.95);
    background:#0D1516;
}
.mock-thumb img{
    width:100%;
    height:100%;
    object-fit:cover;
    object-position:center;
    display:block;
}
.mock-step-bottom{
    display:flex;
    align-items:flex-start;
    gap:.62rem;
    margin-top:0;
    flex:1;
}
.mock-num{
    width:33px;
    height:33px;
    flex:0 0 33px;
    display:grid;
    place-items:center;
    border-radius:11px;
    background:#FFFFFF;
    border:1px solid #D7E3DF;
    color:#315F56;
    font-size:.60rem;
    font-weight:800;
}
.step-b .mock-num{color:#7C6B4E}
.step-c .mock-num{color:#3E796C}
.step-d .mock-num{color:#4B6D84}
.mock-step b{
    display:block;
    color:#304345;
    font-size:.78rem;
    line-height:1.18;
}
.mock-step small{
    display:block;
    margin-top:.24rem;
    color:#7E8E90;
    font-size:.60rem;
    line-height:1.42;
}
.mock-arrow{
    width:29px;
    height:29px;
    display:grid;
    place-items:center;
    border-radius:50%;
    background:#FFFFFF;
    border:1px solid #DCE6E3;
    color:#6F8F86;
    font-size:1rem;
    box-shadow:0 5px 11px rgba(32,57,52,.03);
}

/* upload heading */
.upload-intro{
    display:none !important;
}
.upload-icon{
    width:48px;
    height:48px;
    margin:0 auto .52rem;
    display:grid;
    place-items:center;
    border-radius:50%;
    background:#E9F3EF;
    color:#315F56;
    border:1px solid #D8E7E1;
    font-size:1.2rem;
    font-weight:700;
}
.upload-title{
    font-family:'Libre Baskerville',Georgia,serif;
    color:#223536;
    font-size:1.55rem;
    line-height:1.05;
}
.upload-sub{
    margin-top:.28rem;
    color:#7C8A8D;
    font-size:.78rem;
}

/* native Streamlit uploader: true centered presentation */
[data-testid="stFileUploader"]{
    margin:0 auto 1rem !important;
    padding:0 !important;
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}
[data-testid="stFileUploaderDropzone"]{
    min-height:170px !important;
    padding:1.2rem 1.4rem !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
    justify-content:center !important;
    text-align:center !important;
    gap:.55rem !important;
    background:
        radial-gradient(circle at 50% 22%,rgba(222,239,233,.30),transparent 22%),
        linear-gradient(180deg,#FFFFFF 0%,#FAFCFB 100%) !important;
    border:1.5px dashed #ABC9BF !important;
    border-radius:24px !important;
    box-shadow:0 13px 32px rgba(31,55,50,.035) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"]{
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
    justify-content:center !important;
    text-align:center !important;
    width:auto !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div{
    text-align:center !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small{
    text-align:center !important;
}
[data-testid="stFileUploader"] p{
    font-size:.94rem !important;
    font-weight:600 !important;
    color:#354749 !important;
}
[data-testid="stFileUploader"] small{
    font-size:.74rem !important;
    color:#7A898B !important;
}
[data-testid="stFileUploader"] button{
    margin:.25rem auto 0 !important;
    background:#274F49 !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:12px !important;
    padding:.72rem 1.1rem !important;
    font-size:.80rem !important;
    font-weight:700 !important;
    box-shadow:none !important;
}

/* actual post-upload cards */
.actual-review{
    margin-top:1.15rem !important;
}
.actual-thumb img{
    object-fit:contain !important;
}

/* responsive */
@media(max-width:1000px){
    .mock-flow{grid-template-columns:1fr 1fr}
    .mock-arrow{display:none}
    .mock-lung{opacity:.35;right:.8rem}
}
@media(max-width:650px){
    .mock-workspace{padding:1.4rem 1.1rem}
    .mock-title{font-size:2.5rem}
    .mock-flow{grid-template-columns:1fr}
    .mock-lung{display:none}
}


/* ===== USER REQUESTED MICRO-FIXES ===== */

/* Lung: slightly higher, no visible white frame */
.mock-lung{
    top:-1.45rem !important;
    right:1.35rem !important;
    width:250px !important;
    height:170px !important;
    opacity:.72 !important;
    overflow:hidden !important;
}
.mock-lung img{
    width:100% !important;
    height:100% !important;
    object-fit:cover !important;
    object-position:center top !important;
    mix-blend-mode:multiply !important;
    border:0 !important;
    outline:0 !important;
    box-shadow:none !important;
}

/* Pre-upload step CTs: full scan visible, not confusing close-up crops */
.mock-thumb{
    width:82px !important;
    height:82px !important;
    flex:0 0 82px !important;
    background:#101718 !important;
}
.mock-thumb img{
    width:100% !important;
    height:100% !important;
    object-fit:contain !important;
    object-position:center !important;
    transform:none !important;
    border:0 !important;
    outline:0 !important;
}

/* Hero CTPA: remove the extra outer/inner white-looking frame */
.ct-card{
    padding:0 !important;
    background:transparent !important;
    border:0 !important;
    box-shadow:0 22px 52px rgba(0,0,0,.22) !important;
    backdrop-filter:none !important;
}
.ct-card-inner{
    border:0 !important;
    outline:0 !important;
    box-shadow:none !important;
    border-radius:19px !important;
}
.ct-card img{
    border:0 !important;
    outline:0 !important;
    box-shadow:none !important;
}

/* Keep the actual post-upload previews compact on the left as well */
.actual-thumb{
    width:78px !important;
    height:78px !important;
    flex:0 0 78px !important;
}
.actual-thumb img{
    object-fit:contain !important;
    object-position:center !important;
}


/* ===== FINAL TYPOGRAPHY PASS ===== */
html, body, .stApp,
button, input, textarea, select,
[data-testid="stFileUploader"],
.info-card, .mini-metric,
.result-ribbon, .benchmark-shell,
.mock-workspace, .mock-step,
.actual-review, .actual-card{
    font-family:'Source Sans 3',Arial,sans-serif !important;
}

.hero-title,
.mock-title,
.actual-title,
.section-title,
.upload-title{
    font-family:'Libre Baskerville',Georgia,serif !important;
    font-weight:700 !important;
    letter-spacing:-.025em !important;
}

/* Make editorial titles feel calmer and more academic */
.hero-title{
    line-height:1.08 !important;
}
.mock-title{
    line-height:1.06 !important;
}
.actual-title{
    line-height:1.08 !important;
}

/* Cleaner body copy */
.hero-subtitle,
.mock-sub,
.actual-note,
.mock-step small,
.actual-meta small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small{
    font-family:'Source Sans 3',Arial,sans-serif !important;
    letter-spacing:0 !important;
}

/* Small labels: clinical, not tech/startup */
.hero-kicker,
.mock-kicker,
.actual-kicker{
    font-family:'Source Sans 3',Arial,sans-serif !important;
    letter-spacing:.14em !important;
    font-weight:700 !important;
}

/* Softer pills */
.hero-tag{
    font-family:'Source Sans 3',Arial,sans-serif !important;
    font-weight:600 !important;
    letter-spacing:0 !important;
}

</style>



    """
_main_style = "\n".join(line.lstrip() for line in _main_style.splitlines() if line.strip())
st.markdown(_main_style, unsafe_allow_html=True)

# =========================================================
# 5. HERO - KEEP YOUR VIDEO
# =========================================================
hero_component = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:transparent;font-family:'Source Sans 3',Arial,sans-serif;overflow:hidden}
.hero{
    position:relative;
    width:100%;
    height:610px;
    overflow:hidden;
    border-radius:0 0 30px 30px;
    background:#0A1716;
    border:1px solid #DDE5E2;
    box-shadow:0 22px 54px rgba(19,43,39,.10);
}
.hero-video{
    position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
    opacity:.63;
    filter:saturate(.58) contrast(1.08) brightness(.60);
}
.hero-shade{
    position:absolute;inset:0;
    background:
      linear-gradient(90deg,rgba(7,21,20,.86) 0%,rgba(7,21,20,.58) 43%,rgba(7,21,20,.22) 72%,rgba(7,21,20,.46) 100%),
      linear-gradient(180deg,rgba(0,0,0,.03),rgba(0,0,0,.18));
}
.i3l-mark{
    position:absolute;z-index:8;left:30px;top:26px;
    width:64px;height:auto;object-fit:contain;
    filter:drop-shadow(0 4px 12px rgba(0,0,0,.18));
}
.gpt-credit{
    position:absolute;z-index:8;right:30px;top:28px;
    display:flex;align-items:center;gap:8px;
    color:rgba(242,248,246,.86);
    font-size:9px;font-weight:600;
}
.gpt-credit img{width:23px;height:23px;object-fit:contain}
.hero-copy{
    position:absolute;z-index:6;
    left:8.3%;
    top:50%;
    transform:translateY(-49%);
    width:52%;
}
.hero-kicker{
    display:inline-flex;align-items:center;gap:9px;
    color:#A8C6BD;
    font-size:10px;font-weight:700;
    letter-spacing:.17em;text-transform:uppercase;
}
.hero-kicker:before{content:"";width:28px;height:1px;background:#7CA69A}
.hero-title{
    margin:18px 0 0;
    color:#F3F6F5;
    font-family:'Libre Baskerville',Georgia,serif;
    font-size:61px;line-height:.97;
    letter-spacing:-.035em;font-weight:400;
}
.hero-subtitle{
    max-width:640px;margin-top:18px;
    color:#C7D3D0;
    font-size:14px;line-height:1.7;font-weight:500;
}
.hero-tags{
    display:flex;gap:8px;flex-wrap:wrap;margin-top:24px;
}
.hero-tag{
    padding:8px 13px;
    border-radius:999px;
    color:#E8EFED;
    font-size:10px;font-weight:700;
    border:1px solid rgba(183,215,205,.38);
    background:rgba(19,52,48,.42);
    backdrop-filter:blur(8px);
}
.ct-card{
    position:absolute;z-index:5;
    right:4.2%;top:17%;
    width:29%;height:66%;
    padding:10px;
    border-radius:22px;
    background:rgba(11,25,24,.72);
    border:1px solid rgba(215,229,225,.23);
    box-shadow:0 24px 60px rgba(0,0,0,.23);
    backdrop-filter:blur(8px);
}
.ct-card-inner{
    position:relative;width:100%;height:100%;
    border-radius:16px;overflow:hidden;
    border:1px solid rgba(255,255,255,.10);
}
.ct-card img{
    width:100%;height:100%;object-fit:cover;
    filter:contrast(1.06) brightness(.92);
}
.ct-card-top{
    position:absolute;z-index:3;left:14px;top:13px;
    padding:6px 9px;border-radius:999px;
    background:rgba(246,249,248,.90);
    color:#355E55;font-size:8px;font-weight:800;letter-spacing:.12em;
}
.ct-card-footer{
    position:absolute;z-index:3;left:14px;right:14px;bottom:12px;
    display:flex;justify-content:space-between;align-items:center;
    color:#D4DFDC;font-size:8px;font-weight:600;
}
.ct-card-footer span:last-child{display:flex;align-items:center;gap:6px}
.ct-card-footer i{
    width:7px;height:7px;border-radius:50%;
    background:#4CB9A3;
    box-shadow:0 0 0 4px rgba(76,185,163,.12);
}
@media(max-width:930px){
    .hero-copy{left:7%;width:86%}
    .ct-card{display:none}
    .hero-title{font-size:50px}
}

/* ===== Final revision fixes ===== */
.ct-card{
    padding:0 !important;
    background:transparent !important;
    border:none !important;
    box-shadow:0 18px 40px rgba(0,0,0,.22) !important;
    backdrop-filter:none !important;
}
.ct-card-inner{
    background:#050909 !important;
    border:1px solid rgba(155,182,177,.20) !important;
    box-shadow:none !important;
    overflow:hidden !important;
}
.ct-card img{
    object-fit:cover !important;
    object-position:center center !important;
    border:none !important;
    outline:none !important;
    box-shadow:none !important;
    background:#050909 !important;
}
.ct-card-top{
    background:rgba(209,216,216,.88) !important;
    color:#49635e !important;
    border:1px solid rgba(70,94,90,.10) !important;
    box-shadow:none !important;
}
.ct-card-footer{
    justify-content:flex-end !important;
    left:12px !important;
    right:12px !important;
}
.ct-card-footer span:first-child{
    display:none !important;
}

.mock-lung{
    top:-1.15rem !important;
    right:1.45rem !important;
    width:225px !important;
    height:150px !important;
    overflow:hidden !important;
    opacity:.76 !important;
}
.mock-lung img{
    width:100% !important;
    height:100% !important;
    object-fit:cover !important;
    object-position:center center !important;
    border:none !important;
    outline:none !important;
    box-shadow:none !important;
    mix-blend-mode:multiply !important;
}


html,body{
    font-family:'Source Sans 3',Arial,sans-serif !important;
}
.hero-title{
    font-family:'Libre Baskerville',Georgia,serif !important;
    font-weight:700 !important;
    letter-spacing:-.03em !important;
}
.hero-kicker,.hero-subtitle,.hero-tag,.gpt-credit,.ct-card-top,.ct-card-footer{
    font-family:'Source Sans 3',Arial,sans-serif !important;
}

</style>
</head>
<body>
<div class="hero">
    <video id="heroVideo" class="hero-video" autoplay muted loop playsinline webkit-playsinline preload="auto">
        <source src="__VIDEO__" type="video/mp4">
    </video>
    <div class="hero-shade"></div>

    <img class="i3l-mark" src="__I3L__" alt="i3L University">
    <div class="gpt-credit">
        <img src="__GPT__" alt="ChatGPT">
        <span>Assisted by ChatGPT</span>
    </div>

    <div class="hero-copy">
        <div class="hero-kicker">CTPA Segmentation Review</div>
        <div class="hero-title">Pulmonary Embolism<br>Segmentation</div>
        <div class="hero-subtitle">
            Review CTPA slices, radiologist annotations, and model outputs
            side by side in one focused workspace.
        </div>
        <div class="hero-tags">
            <span class="hero-tag">CTPA</span>
            <span class="hero-tag">Segmentation</span>
            <span class="hero-tag">Ground Truth</span>
            <span class="hero-tag">YOLOv11</span>
            <span class="hero-tag">YOLOv26</span>
        </div>
    </div>

    <div class="ct-card">
        <div class="ct-card-inner">
            <img src="__CT__" alt="CTPA segmentation preview">
            <div class="ct-card-top">CTPA</div>
            <div class="ct-card-footer">
                <span>Axial CTPA slice</span>
                <span><i></i> Segmentation overlay</span>
            </div>
        </div>
    </div>
</div>

<script>
const v=document.getElementById('heroVideo');
v.muted=true;v.defaultMuted=true;v.loop=true;v.playsInline=true;
const playVideo=()=>{
    try{
        const p=v.play();
        if(p&&typeof p.catch==="function"){
            p.catch(()=>setTimeout(()=>{v.muted=true;v.play().catch(()=>{})},250));
        }
    }catch(e){}
};
v.addEventListener('loadedmetadata',playVideo);
v.addEventListener('canplay',playVideo);
v.addEventListener('ended',()=>{try{v.currentTime=0}catch(e){};playVideo()});
v.addEventListener('pause',()=>{if(!document.hidden)setTimeout(playVideo,220)});
v.addEventListener('stalled',()=>setTimeout(playVideo,450));
document.addEventListener('visibilitychange',()=>{if(!document.hidden)playVideo()});
window.addEventListener('focus',playVideo);
setInterval(()=>{if(v.paused||v.ended)playVideo()},1200);
setTimeout(playVideo,120);
</script>
</body>
</html>
"""
hero_component = hero_component.replace("__VIDEO__", CLOUDINARY_VIDEO_URL)
hero_component = hero_component.replace("__CT__", "data:image/jpeg;base64,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")
hero_component = hero_component.replace("__I3L__", "data:image/png;base64,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")
hero_component = hero_component.replace("__GPT__", "data:image/png;base64,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")
components.html(hero_component, height=630, scrolling=False)


# Prevent indented HTML strings from rendering as Markdown code blocks.
def render_html(markup, **kwargs):
    compact = " ".join(str(markup).split())
    st.markdown(
        compact,
        unsafe_allow_html=True,
        **{k: v for k, v in kwargs.items() if k != "unsafe_allow_html"},
    )


# Prevent indented HTML strings from rendering as Markdown code blocks.
def render_html(markup, **kwargs):
    compact = " ".join(str(markup).split())
    st.markdown(compact, unsafe_allow_html=True, **{k: v for k, v in kwargs.items() if k != "unsafe_allow_html"})
# =========================================================
# 6. BENCHMARK INTRO + UPLOAD
# =========================================================








render_html(
    """
    <section class="mock-workspace">
        <div class="mock-workspace-head">
            <div class="mock-workspace-copy">
                <div class="mock-kicker">Case review</div>
                <div class="mock-title">CTPA Segmentation Review</div>
                <div class="mock-sub">
                    Upload a CTPA slice to review the original image, radiologist annotation,
                    and both model segmentations side by side.
                </div>
            </div>
            <div class="mock-lung">
                <img src="data:image/jpeg;base64,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" alt="Lung illustration">
            </div>
        </div>

        <div class="mock-flow">
            <div class="mock-step step-a">
                <div class="mock-thumb"><img src="data:image/jpeg;base64,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" alt="Original CTPA preview"></div>
                <div class="mock-step-bottom">
                    <span class="mock-num">01</span>
                    <div><b>Original CTPA</b><small>View the uploaded CTPA slice.</small></div>
                </div>
            </div>

            <div class="mock-arrow">›</div>

            <div class="mock-step step-b">
                <div class="mock-thumb"><img src="data:image/jpeg;base64,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" alt="Radiologist GT preview"></div>
                <div class="mock-step-bottom">
                    <span class="mock-num">02</span>
                    <div><b>Radiologist GT</b><small>Reference annotation by radiologist.</small></div>
                </div>
            </div>

            <div class="mock-arrow">›</div>

            <div class="mock-step step-c">
                <div class="mock-thumb"><img src="data:image/jpeg;base64,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" alt="YOLOv11 preview"></div>
                <div class="mock-step-bottom">
                    <span class="mock-num">03</span>
                    <div><b>YOLOv11</b><small>Model segmentation output.</small></div>
                </div>
            </div>

            <div class="mock-arrow">›</div>

            <div class="mock-step step-d">
                <div class="mock-thumb"><img src="data:image/jpeg;base64,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" alt="YOLOv26 preview"></div>
                <div class="mock-step-bottom">
                    <span class="mock-num">04</span>
                    <div><b>YOLOv26</b><small>Model segmentation output.</small></div>
                </div>
            </div>
        </div>
    </section>

    <div class="upload-intro">
        <div class="upload-icon">⇧</div>
        <div class="upload-title">Upload a CTPA Slice</div>
        <div class="upload-sub">Drag and drop an image file here, or browse from your device.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Upload CTPA image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)
# =========================================================
# 7. WAIT FOR INPUT
# =========================================================
if uploaded is None:
    st.stop()
raw_image = Image.open(uploaded).convert("RGB")
# =========================================================
# 8. GT
# =========================================================
label_path = find_matching_label(uploaded.name)
if label_path is not None:
    gt_mask = yolo_seg_txt_to_mask(label_path, raw_image.size)
    gt_image = overlay_mask(
        raw_image,
        gt_mask,
        color=(70, 220, 110),
        alpha=0.70,
    )
else:
    gt_mask = None
    gt_image = raw_image.copy()
# =========================================================
# 9. YOLOv11
# =========================================================
result11 = model11.predict(
    raw_image,
    imgsz=512,
    conf=0.25,
    iou=0.7,
    retina_masks=True,
    verbose=False,
)[0]
mask11 = result_to_binary_mask(result11, raw_image.size)
image11 = overlay_mask(
    raw_image,
    mask11,
    color=(235, 75, 75),
    alpha=0.70,
)
conf11 = get_best_confidence(result11)
det11 = len(result11.boxes) if result11.boxes is not None else 0
iou11 = calculate_iou(gt_mask, mask11) if gt_mask is not None else None
# =========================================================
# 10. YOLOv26 trained segmentation checkpoint
# =========================================================
result26 = model26.predict(
    raw_image,
    imgsz=512,
    conf=0.25,
    iou=0.7,
    retina_masks=True,
    verbose=False,
)[0]
mask26 = result_to_binary_mask(result26, raw_image.size)
image26 = overlay_mask(
    raw_image,
    mask26,
    color=(55, 205, 220),
    alpha=0.70,
)
conf26 = get_best_confidence(result26)
det26 = len(result26.boxes) if result26.boxes is not None else 0
iou26 = calculate_iou(gt_mask, mask26) if gt_mask is not None else None
# Mask diagnostics: never silently present the raw image as a valid mask.
mask_stats = {
    "Ground Truth": None if gt_mask is None else int(np.count_nonzero(gt_mask)),
    "YOLOv11": int(np.count_nonzero(mask11)),
    "YOLOv26": int(np.count_nonzero(mask26)),
}
if label_path is None:
    st.warning(f"GT label .txt not found for '{uploaded.name}'. Expected in: {LABEL_DIR}")
elif mask_stats["Ground Truth"] == 0:
    st.warning(f"GT label found ({label_path.name}), but no polygon pixels were parsed. Check label format/class and whether this slice is negative.")
for name in ("YOLOv11", "YOLOv26"):
    if mask_stats[name] == 0:
        st.warning(f"{name}: no segmentation mask for this image (possibly no PE predicted at confidence 0.25).")
if SHOW_DEBUG:
    with st.expander("Mask diagnostics", expanded=False):
        st.write("Active weights:", {"YOLOv11": YOLO11_PATH.name, "YOLOv26": YOLO26_PATH.name})
        st.write("Mask pixels:", mask_stats)
        st.write("Detection counts:", {"YOLOv11": det11, "YOLOv26": det26})
        st.caption("A detection box does not guarantee a non-empty segmentation polygon. If GT is absent, add a matching labels/<uploaded_filename>.txt file.")
        st.write("GT label:", str(label_path) if label_path else "Not found")

raw_preview_uri = pil_to_data_uri(raw_image)
gt_preview_uri = pil_to_data_uri(gt_image)
y11_preview_uri = pil_to_data_uri(image11)
y26_preview_uri = pil_to_data_uri(image26)

render_html(
    f"""
    <section class="actual-review">
        <div class="actual-review-head">
            <div>
                <div class="actual-kicker">Case preview</div>
                <div class="actual-title">Four aligned views</div>
            </div>
            <div class="actual-note">Same CTPA slice · direct comparison</div>
        </div>

        <div class="actual-flow">
            <div class="actual-card">
                <div class="actual-thumb">
                    <img src="{raw_preview_uri}" alt="Original CTPA">
                </div>
                <div class="actual-meta">
                    <span class="actual-num">01</span>
                    <div>
                        <b>Original CTPA</b>
                        <small>Uploaded source image</small>
                    </div>
                </div>
            </div>

            <div class="actual-arrow">›</div>

            <div class="actual-card actual-gt">
                <div class="actual-thumb">
                    <img src="{gt_preview_uri}" alt="Radiologist ground truth">
                </div>
                <div class="actual-meta">
                    <span class="actual-num">02</span>
                    <div>
                        <b>Radiologist GT</b>
                        <small>Mask parsed from the matching label file</small>
                    </div>
                </div>
            </div>

            <div class="actual-arrow">›</div>

            <div class="actual-card actual-y11">
                <div class="actual-thumb">
                    <img src="{y11_preview_uri}" alt="YOLOv11 segmentation">
                </div>
                <div class="actual-meta">
                    <span class="actual-num">03</span>
                    <div>
                        <b>YOLOv11</b>
                        <small>Predicted segmentation</small>
                    </div>
                </div>
            </div>

            <div class="actual-arrow">›</div>

            <div class="actual-card actual-y26">
                <div class="actual-thumb">
                    <img src="{y26_preview_uri}" alt="YOLOv26 segmentation">
                </div>
                <div class="actual-meta">
                    <span class="actual-num">04</span>
                    <div>
                        <b>YOLOv26</b>
                        <small>Predicted segmentation</small>
                    </div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 11. MAIN BENCHMARK ROW
#     4 IMAGES + COMPACT INFO SIDEBAR
# =========================================================
render_html(
    """

<div class="result-ribbon scroll-reveal scroll-reveal-delay-2"> <div class="result-ribbon-left"> <div class="result-ribbon-icon">CT</div> <div class="result-ribbon-text"> <b>Segmentation Comparison</b> <span>Radiologist annotation and both model outputs are aligned for direct visual comparison.</span> </div> </div> <div class="result-ribbon-right"> <span>Raw</span> <span>GT</span> <span>YOLOv11</span> <span>YOLOv26</span> </div> </div>



    """,
    unsafe_allow_html=True,
)
render_html('<div class="benchmark-shell scroll-reveal scroll-reveal-delay-2">', unsafe_allow_html=True)
raw_col, gt_col, y11_col, y26_col, info_col = st.columns(
    [1, 1, 1, 1, 0.9],
    gap="small",
)
with raw_col:
    render_html('<div class="stagger-1"><div class="image-label"><span class="dot dot-raw"></span>Raw Image</div></div>', unsafe_allow_html=True)
    st.image(raw_image, use_column_width=True)
with gt_col:
    render_html('<div class="stagger-2"><div class="image-label"><span class="dot dot-gt"></span>GT Mask</div></div>', unsafe_allow_html=True)
    st.image(gt_image, use_column_width=True)
with y11_col:
    render_html('<div class="stagger-3"><div class="image-label"><span class="dot dot-11"></span>YOLOv11</div></div>', unsafe_allow_html=True)
    st.image(image11, use_column_width=True)
with y26_col:
    render_html('<div class="stagger-4"><div class="image-label"><span class="dot dot-26"></span>YOLOv26</div></div>', unsafe_allow_html=True)
    st.image(image26, use_column_width=True)
with info_col:
    render_html(
        f"""
        <div class="info-card slide-right">
            <div class="model-head"><div class="model-badge">YOLOv11</div><div class="model-status">Segmentation</div></div>
            <div class="mini-grid">
                <div class="mini-metric">
                    <div class="mini-value">{f"{iou11:.3f}" if iou11 is not None else "N/A"}</div>
                    <div class="mini-label">IoU</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value">{f"{conf11*100:.1f}%" if conf11 is not None else "N/A"}</div>
                    <div class="mini-label">Confidence</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value">{det11}</div>
                    <div class="mini-label">Detections</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_html(
        f"""
        <div class="info-card slide-right">
            <div class="model-head"><div class="model-badge">YOLOv26</div><div class="model-status">Segmentation</div></div>
            <div class="mini-grid">
                <div class="mini-metric">
                    <div class="mini-value">{f"{iou26:.3f}" if iou26 is not None else "N/A"}</div>
                    <div class="mini-label">IoU</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value">{f"{conf26*100:.1f}%" if conf26 is not None else "N/A"}</div>
                    <div class="mini-label">Confidence</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value">{det26}</div>
                    <div class="mini-label">Detections</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_html(
        """

<div class="legend-card"> <div class="legend-item"><span class="legend-dot" style="background:#62D48A"></span>Ground Truth</div> <div class="legend-item"><span class="legend-dot" style="background:#E96666"></span>YOLOv11</div> <div class="legend-item"><span class="legend-dot" style="background:#55CBDC"></span>YOLOv26</div> <div class="legend-item"><span class="legend-dot" style="background:#BBC4CA"></span>Raw CTPA</div> </div>



        """,
        unsafe_allow_html=True,
    )
render_html('</div>', unsafe_allow_html=True)
render_html(
    """

<div class="explain-strip scroll-reveal scroll-reveal-delay-3"> <div class="explain-item"> <b>IoU</b> <span>How closely the predicted mask overlaps with the radiologist annotation.</span> </div> <div class="explain-item"> <b>Confidence</b> <span>The model’s confidence score for the detected region.</span> </div> </div>



    """,
    unsafe_allow_html=True,
)
render_html(
    """
    <div class="research-credit">
        <div class="research-credit-left">
            <img class="i3l-logo" src="data:image/png;base64,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" alt="i3L University">
            <div>
                <strong>Indonesia International Institute for Life Sciences</strong>
                <span>Academic research interface</span>
            </div>
        </div>
        <div class="research-credit-right">
            <img class="gpt-logo" src="data:image/png;base64,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" alt="ChatGPT">
            <div>
                <strong>Interface development assisted by ChatGPT</strong>
                <span>Design and implementation support</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
