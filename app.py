"""
FreshOrNot — Produce Intelligence
Streamlit app powered by MobileNetV2 fine-tuned on the
Swoyam2609 Fresh-and-Stale Classification dataset.

Place your trained model at:  model/freshor_not.pt
If no model is found, the app falls back to a pixel-heuristic analyser.
"""

import os
import datetime
import numpy as np
import streamlit as st
from PIL import Image

# ── Optional inference backends (PyTorch preferred, TF fallback) ─────────────
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Produce profiles ──────────────────────────────────────────────────────────
PRODUCE_PROFILES = {
    "apple":       {"fresh_max": 10, "stale_threshold": 4},
    "banana":      {"fresh_max": 7,  "stale_threshold": 3},
    "tomato":      {"fresh_max": 8,  "stale_threshold": 3},
    "strawberry":  {"fresh_max": 5,  "stale_threshold": 2},
    "broccoli":    {"fresh_max": 7,  "stale_threshold": 3},
    "carrot":      {"fresh_max": 14, "stale_threshold": 5},
    "lettuce":     {"fresh_max": 7,  "stale_threshold": 2},
    "mango":       {"fresh_max": 6,  "stale_threshold": 2},
    "orange":      {"fresh_max": 14, "stale_threshold": 5},
    "pepper":      {"fresh_max": 10, "stale_threshold": 4},
    "bittergourd": {"fresh_max": 5,  "stale_threshold": 2},
    "capsicum":    {"fresh_max": 10, "stale_threshold": 4},
    "cucumber":    {"fresh_max": 7,  "stale_threshold": 3},
    "okra":        {"fresh_max": 5,  "stale_threshold": 2},
    "potato":      {"fresh_max": 21, "stale_threshold": 7},
}

SWOYAM_CLASSES = [
    "freshapples", "freshbanana", "freshbittergroud", "freshcapsicum",
    "freshcucumber", "freshokra", "freshoranges", "freshpotato", "freshtomato",
    "rottenapples", "rottenbanana", "rottenbittergroud", "rottencapsicum",
    "rottencucumber", "rottenokra", "rottenoranges", "rottenpotato", "rottentomato",
]

CLASS_TO_PRODUCE = {
    "freshapples":       "apple",     "rottenapples":       "apple",
    "freshbanana":       "banana",    "rottenbanana":       "banana",
    "freshbittergroud":  "bittergourd","rottenbittergroud": "bittergourd",
    "freshcapsicum":     "capsicum",  "rottencapsicum":     "capsicum",
    "freshcucumber":     "cucumber",  "rottencucumber":     "cucumber",
    "freshokra":         "okra",      "rottenokra":         "okra",
    "freshoranges":      "orange",    "rottenoranges":      "orange",
    "freshpotato":       "potato",    "rottenpotato":       "potato",
    "freshtomato":       "tomato",    "rottentomato":       "tomato",
}

MODEL_PATH_PT = os.path.join(os.path.dirname(__file__), "model", "freshor_not.pt")
MODEL_PATH_H5 = os.path.join(os.path.dirname(__file__), "model", "freshor_not.h5")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreshOrNot",
    page_icon="🥬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS: outer desktop bg + phone shell ──────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

  /* ── Desktop background ── */
  html, body {
    background-color: #97a6c3 !important;
    margin: 0;
    padding: 0;
  }

  [data-testid="stAppViewContainer"] {
    background: #97a6c3 !important;
    min-height: 100vh !important;
  }

  [data-testid="stHeader"] { display: none !important; }
  [data-testid="stSidebar"] { display: none !important; }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Phone shell ── */
  [data-testid="stMain"] {
    width: 390px !important;
    min-width: 390px !important;
    max-width: 390px !important;
    height: calc(100vh - 40px) !important;
    min-height: unset !important;
    background: #ffffff !important;
    border-radius: 50px !important;
    border: 8px solid #1c1c1c !important;
    box-shadow:
      0 0 0 1px #000,
      0 0 0 3px #2a2a2a,
      0 50px 100px rgba(0,0,0,0.9),
      inset 0 0 0 1px #111 !important;
    overflow: hidden !important;
    padding: 0 !important;
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
  }

  /* Phone side buttons (decorative) */
  [data-testid="stMain"]::before {
    content: '';
    position: absolute;
    right: -11px;
    top: 120px;
    width: 4px;
    height: 60px;
    background: #97a6c3;
    border-radius: 0 3px 3px 0;
    box-shadow: 0 80px 0 #97a6c3;
  }
  [data-testid="stMain"]::after {
    content: '';
    position: absolute;
    left: -11px;
    top: 100px;
    width: 4px;
    height: 35px;
    background: #97a6c3;
    border-radius: 3px 0 0 3px;
    box-shadow: 0 50px 0 #97a6c3, 0 95px 0 #97a6c3;
  }

  /* ── Inner scroll container ── */
  [data-testid="stMainBlockContainer"] {
    padding: 0 18px 30px 18px !important;
    max-width: 390px !important;
    overflow-y: auto !important;
    height: calc(100vh - 134px) !important;
    max-height: calc(100vh - 134px) !important;
    box-sizing: border-box !important;
  }

  /* ── Typography ── */
  * { font-family: 'DM Mono', 'Courier New', monospace; color: #000; }
  h1 { font-family: 'Syne', sans-serif !important; color: #000 !important; }
  h2, h3 { font-family: 'Syne', sans-serif !important; color: #000 !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: #7fff00 !important;
    color: #000 !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 600;
    letter-spacing: 2px;
    border: none !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    width: 100% !important;
  }
  .stButton > button:hover { background: #9fe830 !important; }

  /* ── Selectbox ── */
  [data-testid="stSelectbox"] > div > div {
    background: #97a6c3 !important;
    border: 1px solid #222 !important;
    color: #000 !important;
    font-family: 'DM Mono', monospace !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #97a6c3 !important;
    border: 1px dashed #333 !important;
    border-radius: 6px !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #97a6c3 !important;
    border-bottom: 1px solid #1a1a1a;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #000 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
  }
  .stTabs [aria-selected="true"] {
    color: #000 !important;
    border-bottom: 2px solid #7fff00 !important;
  }

  /* ── Metric boxes ── */
  [data-testid="metric-container"] {
    background: #97a6c3 !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 6px !important;
    padding: 10px !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div > div { background: #7fff00 !important; }

  /* ── Dividers ── */
  hr { border-color: #1e1e1e !important; }

  /* ── Caption ── */
  .stCaption { color: #000 !important; font-size: 9px !important; letter-spacing: 2px !important; }

  /* ── Custom badges / boxes ── */
  .fresh-badge {
    display: inline-block;
    background: #97a6c3;
    border: 1px solid #7fff0040;
    color: #000;
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    padding: 12px 24px;
    border-radius: 6px;
    letter-spacing: -1px;
    text-shadow: 0 0 20px #7fff0060;
  }
  .stale-badge {
    display: inline-block;
    background: #97a6c3;
    border: 1px solid #ff3b3b40;
    color: #000;
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    padding: 12px 24px;
    border-radius: 6px;
    letter-spacing: -1px;
    text-shadow: 0 0 20px #ff3b3b60;
  }
  .meta-label {
    font-size: 9px;
    color: #000;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .action-box {
    padding: 12px 16px;
    border-radius: 6px;
    background: #97a6c3;
    border: 1px solid #2a2a2a;
    font-size: 12px;
    letter-spacing: 1px;
    font-weight: 500;
  }
  .model-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #111;
    font-size: 11px;
  }
  .model-key { color: #000; letter-spacing: 1px; font-size: 9px; }
  .model-val { color: #000; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if TORCH_AVAILABLE and os.path.exists(MODEL_PATH_PT):
        try:
            model = torch.load(MODEL_PATH_PT, map_location="cpu", weights_only=False)
            model.eval()
            return ("pt", model)
        except Exception as e:
            st.warning(f"Could not load .pt model: {e}")
    if TF_AVAILABLE and os.path.exists(MODEL_PATH_H5):
        try:
            m = tf.keras.models.load_model(MODEL_PATH_H5)
            return ("h5", m)
        except Exception as e:
            st.warning(f"Could not load .h5 model: {e}")
    return None


# ── Inference helpers ─────────────────────────────────────────────────────────
_PT_TRANSFORM = None

def _get_pt_transform():
    global _PT_TRANSFORM
    if _PT_TRANSFORM is None and TORCH_AVAILABLE:
        _PT_TRANSFORM = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _PT_TRANSFORM

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_pt(img: Image.Image):
    return _get_pt_transform()(img.convert("RGB")).unsqueeze(0)

def _shelf_days(is_fresh: bool, score: float, profile: dict) -> int:
    if is_fresh:
        return max(0, round(
            profile["stale_threshold"] +
            (profile["fresh_max"] - profile["stale_threshold"]) * score
        ))
    return max(0, round(profile["stale_threshold"] * (score / 0.42)))

def heuristic_inference(arr: np.ndarray) -> dict:
    img = arr[0]
    r, g, b = float(np.mean(img[:, :, 0])), float(np.mean(img[:, :, 1])), float(np.mean(img[:, :, 2]))
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    saturation = (max_c - min_c) / (max_c + 1e-6)
    brightness = (r + g + b) / 3.0
    fresh_score = saturation * 0.55 + brightness * 0.45
    noise = (int(r * 17 + g * 31 + b * 13) % 23) / 100.0 - 0.115
    final = min(1.0, max(0.0, fresh_score + noise))
    is_fresh = final > 0.42
    conf = min(0.98, (0.72 + final * 0.25) if is_fresh else (0.68 + (1 - final) * 0.28))
    profile = {"fresh_max": 8, "stale_threshold": 3}
    return {
        "label": "FRESH" if is_fresh else "STALE",
        "confidence": conf,
        "shelf_days": _shelf_days(is_fresh, final, profile),
        "fresh_score": final,
        "produce": "unknown",
        "source": "Pixel Heuristic (no model loaded)",
    }

def _parse_preds(preds_array) -> tuple[bool, float, str]:
    n = preds_array.shape[-1]
    detected_produce = "unknown"
    if n == 1:
        raw = float(preds_array.flat[0])
        is_fresh = raw >= 0.5
        conf = raw if is_fresh else 1.0 - raw
    elif n == 2:
        is_fresh = preds_array[0][0] >= preds_array[0][1]
        conf = float(np.max(preds_array[0]))
    else:
        idx = int(np.argmax(preds_array[0]))
        conf = float(preds_array[0][idx])
        label_name = SWOYAM_CLASSES[idx] if idx < len(SWOYAM_CLASSES) else ""
        is_fresh = label_name.startswith("fresh")
        detected_produce = CLASS_TO_PRODUCE.get(label_name, "unknown")
    return is_fresh, conf, detected_produce

def model_inference(model_tuple, img: Image.Image) -> dict:
    backend, model = model_tuple
    if backend == "pt":
        import torch
        tensor = preprocess_pt(img)
        with torch.no_grad():
            out = model(tensor)
        if isinstance(out, (list, tuple)):
            fresh_logit = out[0].numpy()
            preds = 1 / (1 + np.exp(-fresh_logit))
            preds = preds.reshape(1, -1)
        else:
            preds = torch.softmax(out, dim=1).numpy() if out.shape[-1] > 1 else \
                    (1 / (1 + np.exp(-out.numpy())))
        is_fresh, conf, detected_produce = _parse_preds(preds)
    else:
        arr = preprocess(img)
        raw = model.predict(arr, verbose=0)
        if isinstance(raw, (list, tuple)):
            preds = raw[0]
        else:
            preds = raw
        is_fresh, conf, detected_produce = _parse_preds(preds)

    score = conf if is_fresh else (1.0 - conf)
    profile = PRODUCE_PROFILES.get(detected_produce, {"fresh_max": 8, "stale_threshold": 3})
    return {
        "label": "FRESH" if is_fresh else "STALE",
        "confidence": conf,
        "shelf_days": _shelf_days(is_fresh, score, profile),
        "fresh_score": score,
        "produce": detected_produce,
        "source": f"MobileNetV2 (Swoyam2609 · {backend.upper()})",
    }

def run_inference(img: Image.Image) -> dict:
    model = load_model()
    if model is not None:
        return model_inference(model, img)
    arr = preprocess(img)
    return heuristic_inference(arr)

def get_action(result: dict):
    label, days = result["label"], result["shelf_days"]
    if label == "STALE":
        return ("REMOVE IMMEDIATELY", "#000") if days == 0 else ("MARKDOWN NOW", "#000")
    return ("MONITOR CLOSELY", "#000") if days <= 3 else ("NO ACTION NEEDED", "#000")

def shelf_bar_html(days: int, max_days: int) -> str:
    pct = min(100, (days / max(max_days, 1)) * 100)
    color = "#7fff00" if days >= 5 else "#ffb700" if days >= 2 else "#ff3b3b"
    return f"""
    <div style="position:relative;height:10px;background:#97a6c3;border-radius:2px;overflow:hidden;margin:4px 0">
      <div style="position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;
                  background:{color};border-radius:2px;
                  box-shadow:0 0 12px {color}88;"></div>
    </div>"""


# ── Phone notch ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="width:100%;display:flex;justify-content:center;padding-top:14px;margin-bottom:2px">
  <div style="width:126px;height:30px;background:#97a6c3;border-radius:0 0 20px 20px;
              display:flex;align-items:center;justify-content:center;gap:8px">
    <div style="width:8px;height:8px;background:#97a6c3;border-radius:50%;
                border:1px solid #2a2a2a"></div>
    <div style="width:44px;height:5px;background:#97a6c3;border-radius:3px"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Status bar ────────────────────────────────────────────────────────────────
now = datetime.datetime.now().strftime("%H:%M")
st.markdown(f"""
<div style="display:flex;justify-content:space-between;padding:4px 16px 0 16px;
            font-size:10px;color:#000;letter-spacing:1px">
  <span>{now}</span>
  <span>● ● ●</span>
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_status = st.columns([3, 1])
with col_logo:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0 4px 0">
      <div style="width:36px;height:36px;background:#7fff00;border-radius:4px;
                  display:flex;align-items:center;justify-content:center;font-size:20px">🥬</div>
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                    letter-spacing:-0.5px;color:#000">FreshOrNot</div>
        <div style="font-size:9px;color:#000;letter-spacing:2px">PRODUCE INTELLIGENCE v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_status:
    model_exists = os.path.exists(MODEL_PATH_PT) or os.path.exists(MODEL_PATH_H5)
    indicator = "🟢" if model_exists else "🟡"
    status_text = "MODEL READY" if model_exists else "HEURISTIC"
    st.markdown(f"""
    <div style="text-align:right;padding-top:14px">
      <div style="font-size:10px;color:#000;letter-spacing:1px">{indicator} {status_text}</div>
      <div style="font-size:9px;color:#000;letter-spacing:1px">LOCAL · NO CLOUD</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
history_label = f"LOG ({len(st.session_state.history)})"
tab_scan, tab_history, tab_model = st.tabs(["SCAN", history_label, "MODEL"])


# ════════════════════════════════════════════════════════════
# SCAN TAB
# ════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown('<div class="meta-label" style="margin-top:12px">CAPTURE / UPLOAD IMAGE</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="On mobile, tap to open camera or photo library",
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)

        if st.button("▶  ANALYZE", use_container_width=True):
            with st.spinner("RUNNING INFERENCE…"):
                result = run_inference(img)

            action_text, action_color = get_action(result)
            detected_produce = result["produce"]
            profile = PRODUCE_PROFILES.get(detected_produce, {"fresh_max": 8, "stale_threshold": 3})

            st.markdown("<hr/>", unsafe_allow_html=True)

            badge_class = "fresh-badge" if result["label"] == "FRESH" else "stale-badge"
            st.markdown(
                f'<div class="{badge_class}">{result["label"]}</div>',
                unsafe_allow_html=True,
            )
            produce_display = detected_produce.upper() if detected_produce != "unknown" else "UNRECOGNISED"
            st.markdown(
                f'<div class="meta-label" style="margin-top:4px">'
                f'DETECTED: <span style="color:#000;letter-spacing:1px">{produce_display}</span></div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="meta-label">CONFIDENCE</div>', unsafe_allow_html=True)
                conf_pct = int(result["confidence"] * 100)
                conf_color = "#000"
                st.markdown(
                    f'<div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;'
                    f'color:{conf_color}">{conf_pct}<span style="font-size:13px;color:#000;'
                    f'font-family:DM Mono,monospace">%</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(result["confidence"])

            with c2:
                st.markdown('<div class="meta-label">SHELF LIFE · HEAD 2</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;'
                    f'color:#000">{result["shelf_days"]}'
                    f'<span style="font-size:13px;color:#000;font-family:DM Mono,monospace"> days</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    shelf_bar_html(result["shelf_days"], profile["fresh_max"]),
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="action-box" style="color:{action_color};margin-top:12px">'
                f'<span style="margin-right:8px">●</span>{action_text}'
                f'<div style="font-size:9px;color:#000;margin-top:4px;font-weight:400">'
                f'RECOMMENDED ACTION</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="font-size:9px;color:#000;letter-spacing:1px;margin-top:8px;'
                f'text-align:right">{result["source"]}</div>',
                unsafe_allow_html=True,
            )

            st.session_state.history.insert(0, {
                "produce": result["produce"],
                "result": result,
                "ts": datetime.datetime.now(),
            })
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#000;border:1px dashed #1e1e1e;
                    border-radius:6px;margin-top:8px">
          <div style="font-size:40px;margin-bottom:12px">📷</div>
          <div style="font-size:11px;letter-spacing:2px">TAP TO CAPTURE / UPLOAD</div>
          <div style="font-size:9px;margin-top:6px;color:#000">JPEG · PNG · WEBP</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HISTORY TAB
# ════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="meta-label">INSPECTION LOG</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#000;font-size:11px;letter-spacing:2px">
          NO SCANS YET
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("CLEAR LOG", use_container_width=False):
            st.session_state.history = []
            st.rerun()

        for entry in st.session_state.history:
            r = entry["result"]
            label_color = "#000"
            profile = PRODUCE_PROFILES.get(entry["produce"], {"fresh_max": 8, "stale_threshold": 3})
            time_str = entry["ts"].strftime("%H:%M:%S")
            st.markdown(f"""
            <div style="background:#97a6c3;border:1px solid #1a1a1a;border-radius:6px;
                        padding:12px;margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:14px;font-weight:600;color:{label_color}">{r["label"]}</span>
                <span style="font-size:9px;color:#000">{time_str}</span>
              </div>
              <div style="font-size:10px;color:#000;margin-bottom:6px">
                {entry["produce"].upper()} · {r["shelf_days"]}d remaining · {int(r["confidence"]*100)}% conf
              </div>
              {shelf_bar_html(r["shelf_days"], profile["fresh_max"])}
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODEL TAB
# ════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="meta-label">MODEL ARCHITECTURE</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#97a6c3;border:1px solid #1a1a1a;border-radius:6px;padding:16px;margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
        <div style="width:3px;height:28px;background:#7fff00;border-radius:2px"></div>
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:#000">
            MobileNetV2 — Fine-Tuned
          </div>
          <div style="font-size:9px;color:#000;letter-spacing:2px">SWOYAM2609 DATASET · RECOMMENDED</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    layers = [
        ("INPUT",      "224×224 RGB Image"),
        ("BACKBONE",   "MobileNetV2 (frozen · ImageNet)"),
        ("TRANSITION", "Conv 1×1, 128 filters (trainable)"),
        ("POOL",       "Global Average Pooling"),
        ("FC NECK",    "Dense 256 → ReLU → Dropout 0.3"),
        ("HEAD 1",     "Dense 1 → Sigmoid → Fresh / Stale"),
        ("HEAD 2",     "Dense 1 → ReLU → Days (0–21)"),
    ]
    for key, desc in layers:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    padding:6px 8px;border-radius:3px;background:#97a6c3;margin-bottom:3px">
          <span style="font-size:9px;color:#000;letter-spacing:2px">{key}</span>
          <span style="font-size:10px;color:#000">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="meta-label">TRAINING CONFIGURATION</div>', unsafe_allow_html=True)
    config_rows = [
        ("DATASET",       "Swoyam2609 Fresh-and-Stale (Kaggle)"),
        ("CLASSES",       "18  (9 fresh + 9 stale produce types)"),
        ("BACKBONE",      "MobileNetV2 (frozen)"),
        ("INPUT SIZE",    "224 × 224 × 3 RGB"),
        ("OPTIMIZER",     "Adam  (lr = 1e-4)"),
        ("HEAD 1 LOSS",   "Binary Cross-Entropy"),
        ("HEAD 2 LOSS",   "Mean Absolute Error"),
        ("AUGMENTATION",  "Flip · Jitter · Random Crop"),
        ("INFERENCE",     "~300 ms CPU  /  ~30 ms GPU"),
    ]
    st.markdown('<div style="background:#97a6c3;border:1px solid #1a1a1a;border-radius:6px;padding:14px">',
                unsafe_allow_html=True)
    for k, v in config_rows:
        st.markdown(
            f'<div class="model-row"><span class="model-key">{k}</span>'
            f'<span class="model-val">{v}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    found_pt = os.path.exists(MODEL_PATH_PT)
    found_h5 = os.path.exists(MODEL_PATH_H5)
    if found_pt:
        st.success(f"✅  PyTorch model loaded: `model/freshor_not.pt`")
    elif found_h5:
        st.success(f"✅  Keras model loaded: `model/freshor_not.h5`")
    else:
        st.warning(
            "⚠️  No model found. Running in **pixel-heuristic fallback mode**.\n\n"
            "To use a real trained model:\n"
            "1. Run `python train.py` to download the Kaggle dataset and train\n"
            "2. The model saves automatically to `model/freshor_not.pt`\n"
            "3. Restart the app — it will load automatically\n\n"
            "**Requirements for training:**  "
            "`pip install kaggle torch torchvision`  "
            "+ Kaggle API key at `~/.kaggle/kaggle.json`"
        )

# ── Home bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:center;padding:16px 0 8px 0">
  <div style="width:120px;height:4px;background:#97a6c3;border-radius:2px"></div>
</div>
""", unsafe_allow_html=True)