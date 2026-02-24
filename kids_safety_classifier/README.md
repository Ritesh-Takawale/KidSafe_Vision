# 🛡️ KidsSafe AI — Image Safety Classifier

A complete Deep Learning application that classifies images as **Safe** or **Unsafe** for children using an Ensemble ML model with Computer Vision feature extraction.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will:
- Generate 3000 synthetic training images (safe & unsafe patterns)
- Extract deep visual features (color histograms, skin tones, edge density, texture, brightness)
- Train a Voting Ensemble: Random Forest + Gradient Boosting + SVM
- Save the model to `models/` directory
- Print accuracy report and cross-validation scores

### 3. Start the Web Application
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 🧠 How It Works

### Feature Extraction Pipeline
The model extracts **multiple complementary features** from each image:

| Feature | Description | Dimensions |
|---------|-------------|------------|
| **Color Histogram** | HSV color distribution (H, S, V channels) | 96 |
| **Skin Tone Ratio** | Percentage of skin-colored pixels | 1 |
| **Edge Features** | Canny edge density in 4 quadrants | 5 |
| **Brightness** | Mean, std, dark/bright pixel ratios | 4 |
| **Texture (Gabor)** | Gabor filter responses at 4 orientations + Laplacian | 9 |
| **Color Moments** | Mean, std, skewness per color channel | 9 |
| **Total** | | **124 features** |

### Model Architecture
```
Voting Ensemble (Soft Voting)
├── Random Forest (200 trees, max_depth=15)
├── Gradient Boosting (150 estimators, lr=0.1)
└── SVM (RBF kernel, C=10)
```

### Safety Indicators
- 🎨 **High color saturation + brightness** → Likely cartoon/safe content
- 🔴 **Excessive dark tones + red hues** → Potential violence
- 🟤 **High skin-tone ratio** → Adult content indicator
- ⚫ **Very dark imagery** → Disturbing content indicator

---

## 📁 Project Structure

```
kids_safety_classifier/
│
├── train_model.py          # Model training script
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── models/                 # Saved ML models
│   ├── classifier.pkl      # Trained ensemble model
│   ├── scaler.pkl          # Feature scaler
│   └── metadata.json       # Model performance metadata
│
├── templates/
│   └── index.html          # Web UI
│
└── static/
    └── uploads/            # Temporary upload storage
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Classify an image (JSON or multipart) |
| `/model-info` | GET | Get model metadata |
| `/health` | GET | Health check |

### Example API Usage
```python
import requests, base64

with open("image.jpg", "rb") as f:
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:5000/predict",
    json={"image": img_b64})

print(response.json())
# {
#   "is_safe": true,
#   "safe_probability": 87.3,
#   "unsafe_probability": 12.7,
#   "confidence": 87.3,
#   "analysis": {
#     "skin_ratio": 8.2,
#     "brightness": 74.1,
#     "edge_density": 23.5,
#     "color_saturation": 61.0
#   }
# }
```

---

## ⚠️ Important Notes

- This model is trained on **synthetic data** as a demonstration. For production use, retrain on real labeled datasets (e.g., NSFW datasets with proper licensing).
- The model uses **heuristic features** (color, skin tone, brightness) — it's an educational proof-of-concept, not a production content moderation system.
- For production child safety systems, consider using cloud APIs (Google Vision SafeSearch, AWS Rekognition, Azure Content Moderator) which are trained on massive real-world datasets.

---

## 🔧 Extending the Model

### Adding Real Training Data
```python
# Place images in:
training_data/
├── safe/        # Safe images for kids
└── unsafe/      # Unsafe/inappropriate images

# Then modify train_model.py to load from these folders
```

### Using with TensorFlow/PyTorch
If you have TensorFlow or PyTorch available, you can replace the sklearn ensemble with a CNN (MobileNet, ResNet) for much higher accuracy on real images.
