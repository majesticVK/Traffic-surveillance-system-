"""
app.py — Stable Traffic Surveillance AI (Chat Fixed)
"""

import streamlit as st
import cv2
import time
import uuid
import torch
from datetime import datetime

from classification.traffic_detector import TrafficDetector
from classification.traffic_classifier import TrafficClassifier
from agents.traffic_agent import TrafficAgent
from memory.event_memory import TrafficEvent
import config

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# =========================
# CONFIG
# =========================
FRAME_SKIP = 3
TARGET_FPS = 10


# =========================
# PAGE
# =========================
st.set_page_config(page_title="Traffic AI", layout="wide")
st.markdown("""
<style>

/* ===== GLOBAL BLACK THEME ===== */
.stApp {
    background: radial-gradient(circle at top, #020617, #000000);
    color: white;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #0B0F14;
}

/* ===== TITLE ===== */
h1 {
    text-align: center;
    color: #00E5FF;
    text-shadow: 0 0 10px #00E5FF;
}

/* ===== GLASS CARDS ===== */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    animation: fadeIn 0.6s ease;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(90deg, #00E5FF, #00FFA3);
    border-radius: 10px;
    color: black;
    font-weight: bold;
    box-shadow: 0 0 15px rgba(0,255,200,0.5);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,255,200,1);
}

/* ===== CHAT ===== */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
    animation: fadeInUp 0.3s ease;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}

@keyframes fadeInUp {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

</style>
""", unsafe_allow_html=True)  


st.markdown("""
<h1>Traffic surveillance system</h1>
<p style='text-align:center; color:gray;'>
Real-time intelligent surveillance & analysis
</p>
""", unsafe_allow_html=True)


# =========================
# INIT SYSTEM
# =========================
@st.cache_resource
def init_system():
    detector = TrafficDetector()
    classifier = TrafficClassifier()
    agent = TrafficAgent()

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(config.BASE_LLM_MODEL)

    try:
        model = PeftModel.from_pretrained(base_model, config.LORA_PATH)
    except Exception:
        model = base_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    return detector, classifier, agent, model, tokenizer, device


detector, classifier, agent, model, tokenizer, device = init_system()


# =========================
# SESSION STATE
# =========================
if "cap" not in st.session_state:
    st.session_state.cap = None

if "video_source" not in st.session_state:
    st.session_state.video_source = None

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Flag so camera loop doesn't rerun while LLM is generating
if "generating" not in st.session_state:
    st.session_state.generating = False


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Input")

mode = st.sidebar.radio("Source", ["Webcam", "Upload"])

uploaded_file = None
if mode == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload MP4", type=["mp4"])

run = st.sidebar.toggle("Run", True)
show_boxes = st.sidebar.toggle("Detections", True)

location = st.sidebar.text_input("Location", config.DEFAULT_LOCATION)

if st.sidebar.button("Clear chat"):
    st.session_state.chat_history = []
    st.rerun()


# =========================
# LOAD SOURCE
# =========================
def load_source():
    if mode == "Webcam":
        if st.session_state.video_source != "webcam":
            if st.session_state.cap:
                st.session_state.cap.release()
            st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            st.session_state.video_source = "webcam"

    elif uploaded_file:
        if st.session_state.video_source != uploaded_file.name:
            if st.session_state.cap:
                st.session_state.cap.release()

            path = f"temp_{uploaded_file.name}"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())

            st.session_state.cap = cv2.VideoCapture(path)
            st.session_state.video_source = uploaded_file.name


load_source()


# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([3, 2])
col1.markdown('<div class="glass">', unsafe_allow_html=True)
video_box = col1.empty()
status_box = col1.empty()
col1.markdown('</div>', unsafe_allow_html=True)
status_box = col1.empty()


# =========================
# LLM RESPONSE
# =========================
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Strip the prompt portion — keep only what the assistant generated
    if "<|assistant|>" in full_text:
        return full_text.split("<|assistant|>")[-1].strip()

    # Fallback: remove the raw prompt text from the front
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()

    return full_text.strip()


# =========================
# CHAT UI  ← rendered BEFORE camera so it doesn't flicker on rerun
# =========================
with col2:

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Assistant")

    chat_area = st.container()

    # Render full history
    with chat_area:
        for user_msg, ai_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(ai_msg)

    # Input box — always visible
    user_input = st.chat_input("Ask about traffic conditions...")

    if user_input:
        # Show user message immediately
        with chat_area:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Generate response with spinner
        with chat_area:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    prompt = agent.build_prompt(user_input)
                    response = generate_response(prompt)
                st.markdown(response)

        # Persist to session state and rerun to stabilise UI
        st.session_state.chat_history.append((user_input, response))
        # Also sync to agent history for context
        agent.history.append((user_input, response))
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# MAIN CAMERA LOOP
# =========================
if run and st.session_state.cap:

    cap = st.session_state.cap
    ret, frame = cap.read()

    if not ret:
        cap.release()
        time.sleep(0.5)
        st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        st.warning("Reconnecting camera...")
        st.stop()

    frame = cv2.resize(frame, (640, 360))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.session_state.frame_count += 1

    # Always show latest frame
    video_box.image(frame_rgb, channels="RGB")

    # Skip heavy processing on most frames
    if st.session_state.frame_count % FRAME_SKIP != 0:
        time.sleep(1 / TARGET_FPS)
        st.rerun()

    now = datetime.now()

    # DETECTION
    detections = detector.detect(frame_rgb)
    vehicle_count = detector.count_vehicles(detections)
    lights = detector.get_traffic_lights(detections)

    light_states = {}
    for i, tl in enumerate(lights):
        light_states[i] = detector.classify_light_color(frame_rgb, tl)

    dominant_light = "unknown"
    if light_states:
        dominant_light = max(set(light_states.values()), key=list(light_states.values()).count)

    obs = {
        "time": now.strftime("%H:%M"),
        "vehicle_count": vehicle_count,
        "light_state": dominant_light,
        "motion": vehicle_count > 0,
        "objects": [d["class"] for d in detections],
        "location": location,
        "vehicle_in_intersection": False,
    }

    result = classifier.classify_observation(obs)

    # Show annotated frame if requested
    if show_boxes:
        annotated = detector.draw_detections(frame_rgb, detections, light_states)
        video_box.image(annotated, channels="RGB")

    # Colour-coded status
    color_map = {
    "NORMAL": "#00FFA3",
    "CONGESTED": "#FFD700",
    "VIOLATION": "#FF4B4B",
    "HIGH_RISK": "#FF0000",
}

color = color_map.get(result["Status"], "#888")

status_box.markdown(f"""
        <div style="
        padding:12px;
        border-radius:12px;
        text-align:center;
        font-weight:bold;
        background: rgba(0,0,0,0.4);
        border: 1px solid {color};
        color: {color};
        box-shadow: 0 0 15px {color};
    ">
    🚦 {result['Status']} · {result['Confidence']}
    <br>
    <span style="font-size:12px; color:#aaa;">{result['Reason']}</span>
    </div>
    """, unsafe_allow_html=True)


 

    # EVENT LOG (throttled — one event per 10 s per status level)
if result["Status"] in ("CONGESTED", "VIOLATION", "HIGH_RISK"):
        last_t = st.session_state.get("last_event_time")
        if not last_t or (now - last_t).seconds >= 10:
            event = TrafficEvent(
                id=str(uuid.uuid4()),
                timestamp=now,
                location=location,
                event_type=result["Status"].lower(),
                confidence=float(result["Confidence"].strip("%")),
                vehicle_count=vehicle_count,
                light_state=dominant_light,
                objects=obs["objects"],
                speed_kmh=0.0,
            )
            agent.memory.add_event(event)
            st.session_state.last_event_time = now


# =========================
# REFRESH LOOP
# =========================
time.sleep(1 / TARGET_FPS)
st.rerun()