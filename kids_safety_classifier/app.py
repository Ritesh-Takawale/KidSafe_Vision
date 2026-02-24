"""
Kids Safety Image Classifier - Flask Web Application
Uses Image Similarity Matching against your safe/unsafe folders
Production-ready for Render deployment
"""

import os
import sys
import json
import numpy as np
import cv2
import gc
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# ─────────────────────────────────────────────
# Image Database Setup
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")

image_database = {
    "safe": [],
    "unsafe": []
}

metadata = {}

# Ensure folders exist
os.makedirs(os.path.join(TRAINING_DATA_DIR, 'safe'), exist_ok=True)
os.makedirs(os.path.join(TRAINING_DATA_DIR, 'unsafe'), exist_ok=True)


# ─────────────────────────────────────────────
# Feature Functions
# ─────────────────────────────────────────────

def compute_image_hash(img, hash_size=16):
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return diff.flatten()


def compute_color_features(img):
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)


def compute_structural_features(img):
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_hist = cv2.calcHist([mag.astype(np.float32)], [0], None, [16], [0, 500])
    mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()

    return np.concatenate([edge_hist, mag_hist])


def compute_all_features(img):
    return np.concatenate([
        compute_color_features(img),
        compute_structural_features(img)
    ])


def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)


def cosine_similarity(feat1, feat2):
    dot = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


# ─────────────────────────────────────────────
# Load Database (IMPORTANT FIX)
# ─────────────────────────────────────────────

def load_image_database():
    global image_database, metadata

    image_database["safe"] = []
    image_database["unsafe"] = []

    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    print("[*] Loading image database...")

    for category in ["safe", "unsafe"]:
        folder = os.path.join(TRAINING_DATA_DIR, category)

        if not os.path.exists(folder):
            print(f"[!] Folder not found: {folder}")
            continue

        files = [
            f for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in supported_ext
        ]

        print(f"[*] Found {len(files)} {category} images")

        for filename in files:
            filepath = os.path.join(folder, filename)
            try:
                img = cv2.imread(filepath)
                if img is None:
                    continue

                img = cv2.resize(img, (128, 128))

                image_database[category].append({
                    "filename": filename,
                    "hash": compute_image_hash(img),
                    "features": compute_all_features(img)
                })

                del img

            except Exception as e:
                print(f"[!] Error loading {filename}: {e}")

    total = len(image_database["safe"]) + len(image_database["unsafe"])

    metadata = {
        "safe_count": len(image_database["safe"]),
        "unsafe_count": len(image_database["unsafe"]),
        "total": total,
        "method": "Image Similarity Matching"
    }

    print(f"[OK] Loaded {total} images")
    return total > 0


# 🔥 LOAD DATABASE ON IMPORT (THIS FIXES RENDER ISSUE)
load_image_database()


# ─────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────

def classify_image(img):
    if not image_database["safe"] and not image_database["unsafe"]:
        return None

    query_hash = compute_image_hash(img)
    query_features = compute_all_features(img)

    best_score = -1
    best_category = None

    for category in ["safe", "unsafe"]:
        for entry in image_database[category]:
            hash_dist = hamming_distance(query_hash, entry["hash"])
            feat_sim = cosine_similarity(query_features, entry["features"])

            score = 1 - (hash_dist / 256) + feat_sim

            if score > best_score:
                best_score = score
                best_category = category

    if best_category is None:
        return None

    return {
        "category": best_category.upper(),
        "confidence": round(best_score * 50, 2),
        "safe_images": len(image_database["safe"]),
        "unsafe_images": len(image_database["unsafe"])
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    db_ready = bool(image_database["safe"] or image_database["unsafe"])
    return render_template('index.html',
                           model_ready=db_ready,
                           metadata=metadata)


@app.route('/predict', methods=['POST'])
def predict():
    if not image_database["safe"] and not image_database["unsafe"]:
        return jsonify({"error": "Database not loaded"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if '.' not in file.filename or \
       file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid file type"}), 400

    try:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        result = classify_image(img)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "database_loaded": bool(image_database["safe"] or image_database["unsafe"]),
        "safe_images": len(image_database["safe"]),
        "unsafe_images": len(image_database["unsafe"])
    })


@app.route('/reload', methods=['POST'])
def reload_database():
    load_image_database()
    return jsonify({"status": "reloaded"})
