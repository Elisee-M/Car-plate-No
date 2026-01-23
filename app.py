import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Load models once
@st.cache_resource
def load_models():
    yolo = YOLO("license-plate-finetune-v1n.pt")
    reader = easyocr.Reader(["en"])
    return yolo, reader

yolo, reader = load_models()

st.title("🚗 License Plate Recognition")

uploaded = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", channels="BGR")

    results = yolo(image)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        plate = image[y1:y2, x1:x2]

        text = reader.readtext(plate, detail=0)
        st.success(f"Detected Plate: {' '.join(text)}")
