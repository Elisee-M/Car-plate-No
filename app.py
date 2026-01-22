from flask import Flask, request, jsonify, render_template   # <- added render_template
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import cv2
import os
import re
import numpy as np
import warnings
import traceback

warnings.filterwarnings("ignore", category=RuntimeWarning)

app = Flask(__name__, template_folder="templates")   # <- explicit folder
CORS(app)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "license-plate-finetune-v1n.pt")

plate_detector = None
ocr_reader = None

def load_models():
    global plate_detector, ocr_reader
    if plate_detector is None:
        print("Loading YOLO model...")
        plate_detector = YOLO(MODEL_PATH)
        print("YOLO model loaded ✅")
    if ocr_reader is None:
        print("Loading EasyOCR reader...")
        ocr_reader = easyocr.Reader(['en'])
        print("EasyOCR reader loaded ✅")

def preprocess_plate(img, scale=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.medianBlur(thresh, 3)
    return thresh

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# ------------- NEW HOME PAGE -------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ------------- JSON HEALTH (optional) -------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "car-plate-recognition"})

# ------------- API -------------
@app.route("/detect", methods=["POST"])
def detect_plate():
    try:
        load_models()
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Cannot decode image"}), 400

        results = plate_detector(image)
        plates_list = []
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_crop = image[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue
            processed = preprocess_plate(plate_crop)
            ocr_result = ocr_reader.readtext(
                processed,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=0
            )
            plate_text = clean_text("".join(ocr_result)) if ocr_result else "UNKNOWN"
            plates_list.append({"plate": plate_text, "box": [x1, y1, x2, y2]})

        return jsonify({"count": len(plates_list), "plates": plates_list})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)