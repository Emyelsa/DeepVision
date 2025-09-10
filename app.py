import streamlit as st
import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from model import PhysicalNN

# Page config
st.set_page_config(layout="wide", page_title="DeepVision", page_icon="🌊")

# Banner at the top
banner = Image.open("assets/logo.png")
st.image(banner, use_container_width=True)

# Centered heading with margin below
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom: 30px;'>
        AI Underwater Image Enhancement & Object Detection
    </h1>
    """,
    unsafe_allow_html=True,
)

# Columns: left (interaction + detection info), right (all images)
left_col, right_col = st.columns([3, 5], gap="medium")

# Modes for radio buttons
modes = ["Enhancement Only", "Detection Only", "Enhancement + Detection"]

with left_col:
    mode = st.radio("Choose Mode", modes, index=0)

    # Load detection model if needed
    yolo_model = None
    if mode in ("Detection Only", "Enhancement + Detection"):
        yolo_path = os.path.join("yolovm_detect", "weights", "best.pt")
        if os.path.exists(yolo_path):
            yolo_model = YOLO(yolo_path)
        else:
            st.error("YOLO model not found!")

    # Load enhancement model if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enh_model = None
    if mode in ("Enhancement Only", "Enhancement + Detection"):
        enh_path = os.path.join("underwater_results/checkpoints_1", "model_best_2842.pth.tar")
        if os.path.exists(enh_path):
            enh_model = PhysicalNN().to(device)
            enh_model = nn.DataParallel(enh_model)
            chk = torch.load(enh_path, map_location=device)
            enh_model.load_state_dict(chk["state_dict"])
            enh_model.eval()
        else:
            st.error("Enhancement model not found!")

    # Upload image
    uploaded = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

    # Prepare detection info placeholder
    detection_info = None

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # Process enhancement if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhanced = img
    if mode in ("Enhancement Only", "Enhancement + Detection") and enh_model is not None:
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = enh_model(tensor)
        arr = out.squeeze().cpu().numpy().transpose(1, 2, 0)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        enhanced = Image.fromarray(arr)

    # Prepare detection if needed
    detection_results = None
    if mode in ("Detection Only", "Enhancement + Detection") and yolo_model is not None:
        frame = np.array(enhanced if mode == "Enhancement + Detection" else img)
        res = yolo_model.predict(frame)[0]
        detection_results = res

        # Prepare detection info text with colors
        if res.boxes and res.boxes.cls.numel() > 0:
            detection_info = []
            for i in range(len(res.boxes.cls)):
                cid = int(res.boxes.cls[i])
                nm = res.names[cid]
                cf = float(res.boxes.conf[i])
                bb = [round(v, 1) for v in res.boxes.xyxy[i].tolist()]
                line = (
                     f"• **{nm}**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                     f"confidence: <span style='color:green;'>{cf:.2f}</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                     f"bbox: <span style='color:red;'>{bb}</span>"
                )
                detection_info.append(line)
        else:
            detection_info = ["No objects detected."]

with right_col:
    if uploaded:
        st.image(img, caption="Original Image")
        if mode in ("Enhancement Only", "Enhancement + Detection") and enhanced is not None:
            st.image(enhanced, caption="Enhanced Image")

        if mode in ("Detection Only", "Enhancement + Detection") and detection_results is not None:
            vis = detection_results.plot()
            st.image(vis, caption="Detected Image")
    else:
        st.markdown(
        "<p style='text-align:center; color:gray; margin-top: 60px; font-style: italic;'>No image uploaded yet.</p>", 
        unsafe_allow_html=True
    )


with left_col:
    if detection_info is not None:
        st.subheader("Detection Results")
        for line in detection_info:
            st.markdown(line, unsafe_allow_html=True)
