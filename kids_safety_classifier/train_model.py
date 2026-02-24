"""
Kids Safety Image Classifier - Flask Inference App
Uses trained ML model (classifier.pkl + scaler.pkl)
"""

import os
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
    model = None
    scaler = None


# ─────────────────────────────────────────────
# Feature Extraction (MUST MATCH TRAINING)
# ─────────────────────────────────────────────

def extract_color_histogram(img, bins=32):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)


def extract_skin_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 20, 70], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    skin_mask = mask1 | mask2

    return np.sum(skin_mask > 0) / (img.shape[0] * img.shape[1])


def extract_edge_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return np.array([edge_density])


def extract_features_from_array(img):
    img = cv2.resize(img, (224, 224))

    color_hist = extract_color_histogram(img)
    skin = np.array([extract_skin_ratio(img)])
    edges = extract_edge_features(img)

    return np.concatenate([color_hist, skin, edges])


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None:
        return jsonify({"error": "Model is offline"})

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file"})

        features = extract_features_from_array(img)
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        is_safe = prediction == 0

        safe_prob = float(probabilities[0]) * 100
        unsafe_prob = float(probabilities[1]) * 100
        confidence = max(safe_prob, unsafe_prob)

        return jsonify({
            # Frontend compatibility
            "label": int(prediction),
            "is_safe": bool(is_safe),

            # Display info
            "category": "SAFE" if is_safe else "UNSAFE",
            "confidence": round(confidence, 2),

            # Probabilities
            "safe_probability": round(safe_prob, 2),
            "unsafe_probability": round(unsafe_prob, 2),

            # Prevent undefined errors in UI
            "similarity": round(confidence, 2),
            "hash_distance": 0,
            "match_type": "ML Model",
            "matched_file": "N/A"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
