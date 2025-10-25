import streamlit as st
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace
import io
st.set_page_config(page_title="EmoCare — Patient Emotion Monitoring", page_icon="🧠", layout="centered")
st.title("🧠 Emocare — Patient Emotion Monitoring")
st.write("Upload a patient's photo and the app will detect the face and predict the emotion.")

st.sidebar.header("Settings")
detector_backend = st.sidebar.selectbox("Face detector backend", ["opencv", "mtcnn", "dlib", "retinaface"], index=0)
enforce_detection = st.sidebar.checkbox("Enforce face detection (fail if no face found)", value=False)
neutral_threshold = st.sidebar.slider("Neutral threshold (probability to consider 'neutral')", 0.0, 1.0, 0.0, 0.01)

uploaded = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"])

def read_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    arr = np.array(img)
    # convert RGB->BGR for OpenCV operations
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_bytes(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

if uploaded:
    try:
        image_bgr = read_image(uploaded)
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Analyzing... (this may take a few seconds on first run)"):
            # DeepFace.analyze returns dict (or list for multiple faces) — ask for emotion only
            result = DeepFace.analyze(
                img_path = image_bgr,
                actions = ['emotion'],
                detector_backend = detector_backend,
                enforce_detection = enforce_detection
            )

        # DeepFace returns list when multiple faces detected; normalize to list
        face_results = result if isinstance(result, list) else [result]
        st.success(f"Detected {len(face_results)} face(s).")

        # If multiple faces, show each
        for i, face in enumerate(face_results, start=1):
            st.markdown(f"### Face {i}")
            # emotion dict
            emotions = face.get("emotion", {})
            dominant = face.get("dominant_emotion", None)
            region = face.get("region", None)  # dict with x,y,w,h for opencv
            st.write(f"**Dominant emotion:** {dominant}")

            # display bar chart for probabilities
            if emotions:
                # sort emotions consistently
                labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]
                data = [emotions.get(l, 0) for l in labels]
                chart_data = {lab: float(val) for lab, val in zip(labels, data)}
                st.bar_chart(chart_data)

            # draw bounding box if region exists
            img_copy = image_bgr.copy()
            if region and all(k in region for k in ("x","y","w","h")):
                x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 2)
                label = str(dominant)
                # put label above box
                cv2.putText(img_copy, label, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), caption=f"Face {i} annotated", use_container_width=True)

            # Textual detailed breakdown
            st.write("**Emotion probabilities:**")
            for k,v in sorted(emotions.items(), key=lambda kv: -kv[1]):
                st.write(f"- {k.capitalize()}: {v:.2f}%")

            # Optional: map probabilities to a single sentence
            if dominant:
                prob = emotions.get(dominant, 0.0)/100.0 if isinstance(next(iter(emotions.values())), (int,float)) else emotions.get(dominant, 0.0)
                if dominant.lower() == "neutral" and prob < neutral_threshold:
                    st.info(f"Dominant: {dominant} (low confidence {prob:.2f}) — consider reviewing manually.")
                else:
                    st.success(f"Predicted emotion: **{dominant.capitalize()}** (confidence {prob:.2f})")
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.info("Try another image or turn off 'Enforce face detection' in the sidebar.")

else:
    st.info("Upload a patient's photo to start. Tip: frontal faces give better results.")

st.markdown("---")
st.caption("Prototype by Anju Sree |AI for Healthcare |2025")