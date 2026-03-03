"""
Factory Operator UI — MG400 Vision Pick System
Streamlit app for camera capture, object detection, and robot pick control.
"""

import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.segmentation import detect 

from robot.transform import transform, robot_pick


# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MG400 Vision Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d0f12;
    color: #c8d0dc;
  }

  .stApp { background-color: #0d0f12; }

  /* Header */
  .hdr {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 24px;
  }
  .hdr-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.45rem;
    color: #e2e8f0;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .hdr-sub {
    font-size: 0.75rem;
    color: #4a5568;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
  }
  .status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }
  .dot-green  { background: #38a169; box-shadow: 0 0 8px #38a169; }
  .dot-yellow { background: #d69e2e; box-shadow: 0 0 8px #d69e2e; }
  .dot-red    { background: #e53e3e; box-shadow: 0 0 8px #e53e3e; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  /* Panel card */
  .panel {
    background: #131720;
    border: 1px solid #1e2a38;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 16px;
  }
  .panel-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #4a90d9;
    text-transform: uppercase;
    margin-bottom: 0px;
    border-left: 3px solid #4a90d9;
    padding-left: 8px;
  }

  /* Mode toggle styling */
  div[data-testid="stRadio"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
  }

  /* Buttons */
  .stButton > button {
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-size: 0.78rem;
    border-radius: 3px;
    border: 1px solid #2d3a4a;
    background: #161d28;
    color: #8bacc8;
    transition: all 0.15s ease;
    width: 100%;
    padding: 10px 0;
  }
  .stButton > button:hover {
    background: #1e2d40;
    border-color: #4a90d9;
    color: #d0e4f7;
  }
  .stButton > button:active {
    background: #4a90d9;
    color: #fff;
  }
  /* Run pick — accent */
  .run-pick-btn > button {
    background: #1a3a1a !important;
    border-color: #38a169 !important;
    color: #68d391 !important;
  }
  .run-pick-btn > button:hover {
    background: #22543d !important;
    color: #9ae6b4 !important;
  }
  .run-pick-btn > button:disabled {
    background: #161d28 !important;
    border-color: #2d3a4a !important;
    color: #3a4a5a !important;
    cursor: not-allowed !important;
  }

  /* Coord display */
  .coord-box {
    background: #0a0d12;
    border: 1px solid #2d3a4a;
    border-radius: 4px;
    padding: 14px 18px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    color: #68d391;
    letter-spacing: 2px;
    text-align: center;
    margin: 10px 0;
  }
  .coord-label {
    font-size: 0.65rem;
    color: #4a5568;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 4px;
  }

  /* Status messages */
  .msg-success {
    background: #0f2a1a;
    border: 1px solid #2f855a;
    border-left: 4px solid #38a169;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #68d391;
    letter-spacing: 1px;
  }
  .msg-error {
    background: #2a0f0f;
    border: 1px solid #822727;
    border-left: 4px solid #e53e3e;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #fc8181;
    letter-spacing: 1px;
  }
  .msg-info {
    background: #0f1e2a;
    border: 1px solid #2b4d6e;
    border-left: 4px solid #4a90d9;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #90cdf4;
    letter-spacing: 1px;
  }
  .msg-warn {
    background: #2a1e0a;
    border: 1px solid #744210;
    border-left: 4px solid #d69e2e;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #f6e05e;
    letter-spacing: 1px;
  }

  /* Log area */
  .log-entry {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #4a6a8a;
    padding: 2px 0;
    border-bottom: 1px solid #0e1520;
  }
  .log-entry span { color: #4a90d9; margin-right: 8px; }

  /* Image container */
  .img-wrapper {
    background: #0a0c10;
    border: 1px solid #1e2a38;
    border-radius: 4px;
    overflow: hidden;
    min-height: 320px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .img-placeholder {
    color: #2d3a4a;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-align: center;
  }

  /* Select boxes */
  div[data-testid="stSelectbox"] select,
  div[data-testid="stSelectbox"] > div > div {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    background: #0d1219 !important;
    border-color: #2d3a4a !important;
    color: #8bacc8 !important;
  }

  /* Metric */
  [data-testid="stMetric"] {
    background: #0f151e;
    border: 1px solid #1e2a38;
    border-radius: 4px;
    padding: 10px 14px;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 1.5px !important;
    color: #4a5568 !important;
    text-transform: uppercase !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #90cdf4 !important;
  }

  /* Hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ──────────────────────────────────────────────────────
defaults = {
    "captured_image": None,
    "annotated_image": None,
    "detected": False,
    "targets": None,
    "last_status": None,   # ("success"|"error"|"info"|"warn", message)
    "log": [],
    "pick_count": 0,
    "detect_count": 0,
    "_last_file_id": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Mock / Stub Functions ───────────────────────────────────────────────────
# Replace these with your real camera & robot integration

def load_image_file(uploaded_file) -> Image.Image:
    """Load a PIL Image from a Streamlit uploaded file object."""
    img = Image.open(uploaded_file).convert("RGB")
    return img


def detect_target(image: Image.Image, color_filter: str, shape_filter: str):
    """
    Simulates detection. Returns (annotated_image, x, y) or raises on failure.
    Replace with your real CV pipeline.
    """
    
    with open("calibration/calibration.json", "r") as f:
        lines = [f.readline().strip() for _ in range(3)]
        # Parse the array
        array_str = '\n'.join(lines)
        array_str = array_str.replace('[', '').replace(']', '')
        rows = [line.split() for line in array_str.split('\n')]
        H = np.array(rows, dtype=float)

        print(f"Loaded homography matrix H:\n{H}")
        print(f"\nMetadata:\n{f.readline()}\n{f.readline()}")
    
    # convert to cv2 image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    obj_pos, img_vis = detect(img_cv, color=str.lower(color_filter) if color_filter != "Any Color" else None, shape=str.lower(shape_filter) if shape_filter != "Any Shape" else None)
    if len(obj_pos) == 0:
        print("====== No target found.")
        return None
    else:
        print(f"====== {len(obj_pos)} targets found. ")
    print(f"Detected object positions in pixel coordinates: {len(obj_pos)} objs. {obj_pos}")
    obj_pos_robot = transform(obj_pos, H)

    img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    image_rgb = Image.fromarray(img_rgb)
    return image_rgb, obj_pos_robot


def mock_run_pick(x, y):
    """Simulate robot pick. Replace with actual MG400 API call."""
    time.sleep(0.6)   # simulate motion time
    return True, f"Pick executed → MovJ({x}, {y}, 50, 0) ✓"


def add_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log.insert(0, (ts, msg))
    if len(st.session_state.log) > 20:
        st.session_state.log = st.session_state.log[:20]


# ─── Header ─────────────────────────────────────────────────────────────────
dot_class = "dot-green" if st.session_state.captured_image is not None else "dot-yellow"
img_status = "Image Loaded" if st.session_state.captured_image is not None else "No Image"
st.markdown(f"""
<div class="hdr">
  <div>
    <div class="hdr-title">🤖 MG400 Vision Control</div>
    <div class="hdr-sub">
      <span class="status-dot {dot_class}"></span>
      {img_status}
      &nbsp;·&nbsp; Picks: {st.session_state.pick_count}
      &nbsp;·&nbsp; Detections: {st.session_state.detect_count}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Layout: Left Controls | Center Image | Right Status ────────────────────
col_ctrl, col_img, col_status = st.columns([1.1, 2.4, 1.3], gap="medium")

# ══════════════════════════ LEFT: Controls ═══════════════════════════════════
with col_ctrl:
    # — Mode —
    st.markdown('<div class="panel"><div class="panel-title">// Operation Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        label="mode",
        options=["📋  PLAN", "⚡  EXECUTE"],
        index=0,
        label_visibility="collapsed",
        key="mode_radio",
    )
    execute_mode = "EXECUTE" in mode
    st.markdown('</div>', unsafe_allow_html=True)

    # — Filters —
    st.markdown('<div class="panel"><div class="panel-title">// Detection Filters</div>', unsafe_allow_html=True)
    color_filter = st.selectbox(
        "Color",
        ["Any Color", "Red", "Blue", "Green", "Yellow", "Purple", "Pink", "Black"],
        key="color_sel",
    )
    shape_filter = st.selectbox(
        "Shape",
        ["Any Shape", "Rectangle", "Square", "Circle", "Triangle"],
        key="shape_sel",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # — Action Buttons —
    st.markdown('<div class="panel"><div class="panel-title">// Actions</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📂  LOAD IMAGE",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="file_uploader",
        label_visibility="collapsed",
        help="Upload JPG / PNG / BMP / TIFF",
    )
    if uploaded is not None:
        # Only reload if this is a newly uploaded file
        if uploaded.file_id != st.session_state.get("_last_file_id"):
            img = load_image_file(uploaded)
            st.session_state.captured_image = img
            st.session_state.annotated_image = None
            st.session_state.detected = False
            st.session_state.targets = []
            st.session_state.last_status = ("info", f"Loaded: {uploaded.name}  ({img.width}×{img.height}px)")
            st.session_state._last_file_id = uploaded.file_id
            add_log(f"Loaded image: {uploaded.name}")
            st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if st.button("🔍  DETECT TARGET", key="btn_detect"):
        if st.session_state.captured_image is None:
            st.session_state.last_status = ("warn", "No image — capture first.")
            add_log("Detect called with no image.")
        else:
            with st.spinner(""):
                try:
                    result = detect_target(
                        st.session_state.captured_image,
                        color_filter, shape_filter
                    )
                    if result is None:
                        st.session_state.detected = False
                        st.session_state.targets = []
                        st.session_state.last_status = ("warn", "No targets found in image.")
                        add_log("Detection OK — no targets.")
                    else:
                        ann, obj_pos_robot = result
                        st.session_state.annotated_image = ann
                        st.session_state.targets = obj_pos_robot   # e.g. [[x1,y1],[x2,y2],...]
                        st.session_state.detected = True
                        st.session_state.detect_count = len(obj_pos_robot)
                        n = len(obj_pos_robot)
                        st.session_state.last_status = (
                            "success",
                            f"{n} target{'s' if n > 1 else ''} detected."
                        )
                        add_log(f"Detection OK → {n} targets: {obj_pos_robot}")
                except Exception as e:
                    st.session_state.detected = False
                    st.session_state.last_status = ("error", f"Detection failed: {e}")
                    add_log(f"Detection FAILED: {e}")
        st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    run_disabled = not (execute_mode and st.session_state.detected)
    pick_label = "▶  RUN PICK" if not run_disabled else "⊘  RUN PICK  [PLAN MODE]" if not execute_mode else "⊘  RUN PICK  [NO TARGET]"

    st.markdown('<div class="run-pick-btn">', unsafe_allow_html=True)
    if st.button(pick_label, key="btn_pick", disabled=run_disabled):
        with st.spinner(""):
            success_count = 0
            for i, pos in enumerate(st.session_state.targets):
                x, y = pos[0], pos[1]
                robot_pick(x, y)
                success_count += 1
                add_log(f"Pick {i+1}/{len(st.session_state.targets)} OK → ({x:.1f}, {y:.1f})")
            st.session_state.pick_count += success_count
            st.session_state.last_status = ("success", f"✓ All {success_count} picks completed.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # — Mode hint —
    if not execute_mode:
        st.markdown('<div class="msg-warn">⚠ PLAN mode — picks disabled. Switch to EXECUTE to run robot.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="msg-info">⚡ EXECUTE mode — robot commands active.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════ CENTER: Image ════════════════════════════════════
with col_img:
    st.markdown('<div class="panel-title">// Vision Feed</div>', unsafe_allow_html=True)

    display_img = st.session_state.annotated_image or st.session_state.captured_image
    if display_img is not None:
        st.image(display_img, use_container_width=True)
        caption = "🔴 ANNOTATED" if st.session_state.annotated_image else "📷 RAW CAPTURE"
        st.markdown(
            f"<div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;"
            f"color:#4a5568;letter-spacing:2px;text-align:right;margin-top:4px;'>{caption}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style="background:#0a0c10;border:1px solid #1e2a38;border-radius:4px;
                    min-height:340px;display:flex;align-items:center;justify-content:center;">
          <div style="color:#2d3a4a;font-family:'Share Tech Mono',monospace;
                      font-size:0.8rem;letter-spacing:2px;text-align:center;line-height:2.2;">
            NO IMAGE<br>
            <span style="font-size:0.6rem;color:#1e2a38;">UPLOAD AN IMAGE TO START</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════ RIGHT: Status & Coords ═══════════════════════════
with col_status:
    # Coordinates
    st.markdown('<div class="panel"><div class="panel-title">// Target Coordinates</div>', unsafe_allow_html=True)
    targets = st.session_state.targets
    if targets:
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #0e1520;">'
            f'<span style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#4a5568;">#{i+1}</span>'
            f'<span style="font-family:Share Tech Mono,monospace;font-size:0.82rem;color:#68d391;">X {pos[0]:.1f}</span>'
            f'<span style="font-family:Share Tech Mono,monospace;font-size:0.82rem;color:#90cdf4;">Y {pos[1]:.1f}</span>'
            f'</div>'
            for i, pos in enumerate(targets)
        )
        st.markdown(
            f'<div style="background:#0a0d12;border:1px solid #2d3a4a;border-radius:4px;'
            f'padding:10px 14px;margin:8px 0;">'
            f'<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#4a5568;'
            f'letter-spacing:2px;margin-bottom:6px;">mm · ROBOT FRAME</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="msg-success">● {len(targets)} TARGET{"S" if len(targets)>1 else ""} LOCKED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;'
            'color:#2d3a4a;text-align:center;padding:20px 0;">NO TARGETS</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Status message
    st.markdown('<div class="panel"><div class="panel-title">// Last Status</div>', unsafe_allow_html=True)
    if st.session_state.last_status:
        kind, msg = st.session_state.last_status
        css_class = {
            "success": "msg-success",
            "error":   "msg-error",
            "info":    "msg-info",
            "warn":    "msg-warn",
        }.get(kind, "msg-info")
        st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="msg-info">System ready.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="panel"><div class="panel-title">// Session Stats</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Picks", st.session_state.pick_count)
    m2.metric("Detects", st.session_state.detect_count)
    st.markdown('</div>', unsafe_allow_html=True)

    # Event log
    st.markdown('<div class="panel"><div class="panel-title">// Event Log</div>', unsafe_allow_html=True)
    if st.session_state.log:
        log_html = "".join(
            f'<div class="log-entry"><span>{ts}</span>{msg}</div>'
            for ts, msg in st.session_state.log[:8]
        )
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#2d3a4a;">No events yet.</div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)