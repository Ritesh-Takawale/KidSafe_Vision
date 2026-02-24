"""
Kids Safety Image Classifier - Model Training Script
Uses OpenCV for feature extraction + Ensemble ML model
"""

import numpy as np
import cv2
import os
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

def extract_color_histogram(img, bins=32):
    """Extract HSV color histogram features."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)


def extract_skin_ratio(img):
    """Detect skin-tone pixel ratio (a key safety indicator)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Multiple skin tone ranges
    lower1 = np.array([0, 20, 70], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    skin_mask = mask1 | mask2
    skin_ratio = np.sum(skin_mask > 0) / (img.shape[0] * img.shape[1])
    return skin_ratio


def extract_edge_features(img):
    """Extract edge density and distribution features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    # Divide into quadrants
    h, w = edges.shape
    q1 = np.sum(edges[:h//2, :w//2] > 0) / (h//2 * w//2)
    q2 = np.sum(edges[:h//2, w//2:] > 0) / (h//2 * w//2)
    q3 = np.sum(edges[h//2:, :w//2] > 0) / (h//2 * w//2)
    q4 = np.sum(edges[h//2:, w//2:] > 0) / (h//2 * w//2)
    return np.array([edge_density, q1, q2, q3, q4])


def extract_brightness_features(img):
    """Extract brightness/exposure features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(gray) / 255.0
    std_bright = np.std(gray) / 255.0
    dark_ratio = np.sum(gray < 50) / gray.size
    bright_ratio = np.sum(gray > 200) / gray.size
    return np.array([mean_bright, std_bright, dark_ratio, bright_ratio])


def extract_texture_features(img):
    """Extract LBP-like texture features via Laplacian."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Gabor filter responses
    gabor_features = []
    for theta in [0, 45, 90, 135]:
        theta_rad = theta * np.pi / 180
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        gabor_features.extend([np.mean(filtered), np.std(filtered)])
    return np.array([laplacian_var / 1000.0] + gabor_features)


def extract_color_moments(img):
    """Extract color moments (mean, std, skewness) for each channel."""
    features = []
    for i in range(3):
        channel = img[:, :, i].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skew = np.mean(((channel - mean) / (std + 1e-6)) ** 3)
        features.extend([mean / 255.0, std / 255.0, skew])
    return np.array(features)


def extract_hog_features(img):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi
    orientation[orientation < 0] += 180
    
    # Create HOG with 9 bins, 8x8 cells
    cell_size = 8
    num_bins = 9
    features = []
    
    for i in range(0, 64, cell_size):
        for j in range(0, 64, cell_size):
            cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
            cell_ori = orientation[i:i+cell_size, j:j+cell_size]
            hist, _ = np.histogram(cell_ori, bins=num_bins, range=(0, 180), weights=cell_mag)
            hist = hist / (np.linalg.norm(hist) + 1e-6)  # Normalize
            features.extend(hist)
    
    return np.array(features)


def extract_spatial_features(img):
    """Extract spatial color distribution features."""
    img = cv2.resize(img, (64, 64))
    features = []
    
    # Divide into 4x4 grid and get color stats per region
    grid_size = 16
    for i in range(0, 64, grid_size):
        for j in range(0, 64, grid_size):
            region = img[i:i+grid_size, j:j+grid_size]
            for c in range(3):
                features.append(np.mean(region[:, :, c]) / 255.0)
                features.append(np.std(region[:, :, c]) / 255.0)
    
    return np.array(features)


def extract_contrast_features(img):
    """Extract local contrast and sharpness features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance (sharpness indicator)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
    
    # Local contrast in different regions
    h, w = gray.shape
    regions = [
        gray[:h//2, :w//2],      # Top-left
        gray[:h//2, w//2:],      # Top-right
        gray[h//2:, :w//2],      # Bottom-left
        gray[h//2:, w//2:],      # Bottom-right
        gray[h//4:3*h//4, w//4:3*w//4]  # Center
    ]
    
    contrasts = [np.std(r) / 255.0 for r in regions]
    
    return np.array([lap_var] + contrasts)


def extract_features(img_path):
    """Extract all features from an image file."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))

    color_hist = extract_color_histogram(img)
    skin = np.array([extract_skin_ratio(img)])
    edges = extract_edge_features(img)
    brightness = extract_brightness_features(img)
    texture = extract_texture_features(img)
    moments = extract_color_moments(img)
    hog = extract_hog_features(img)
    spatial = extract_spatial_features(img)
    contrast = extract_contrast_features(img)

    return np.concatenate([color_hist, skin, edges, brightness, texture, moments, hog, spatial, contrast])


def extract_features_from_array(img_array):
    """Extract features from a numpy image array (BGR)."""
    img = cv2.resize(img_array, (224, 224))
    color_hist = extract_color_histogram(img)
    skin = np.array([extract_skin_ratio(img)])
    edges = extract_edge_features(img)
    brightness = extract_brightness_features(img)
    texture = extract_texture_features(img)
    moments = extract_color_moments(img)
    hog = extract_hog_features(img)
    spatial = extract_spatial_features(img)
    contrast = extract_contrast_features(img)
    return np.concatenate([color_hist, skin, edges, brightness, texture, moments, hog, spatial, contrast])


# ─────────────────────────────────────────────
# Load Training Data from Your Dataset
# ─────────────────────────────────────────────

def augment_image(img, aggressive=True):
    """Create augmented versions of an image to expand the dataset."""
    augmented = [img]
    
    # Horizontal flip
    augmented.append(cv2.flip(img, 1))
    
    # Brightness variations (more ranges)
    for alpha in [0.6, 0.75, 0.85, 1.15, 1.3, 1.5]:
        bright = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        augmented.append(bright)
    
    # Rotations (more angles)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-20, -10, 10, 20]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)
    
    # Add noise variations
    for noise_level in [8, 15, 25]:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        augmented.append(noisy)
    
    if aggressive:
        # Contrast adjustments
        for alpha, beta in [(1.2, 10), (0.8, -10), (1.4, 20)]:
            adjusted = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            augmented.append(adjusted)
        
        # Gaussian blur (simulates low quality)
        for ksize in [3, 5]:
            blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
            augmented.append(blurred)
        
        # Color jitter - shift hue/saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        for h_shift in [-10, 10]:
            hsv_shifted = hsv.copy()
            hsv_shifted[:, :, 0] = (hsv_shifted[:, :, 0] + h_shift) % 180
            jittered = cv2.cvtColor(hsv_shifted.astype(np.uint8), cv2.COLOR_HSV2BGR)
            augmented.append(jittered)
        
        # Scale variations (zoom in/out)
        for scale in [0.85, 1.15]:
            scaled = cv2.resize(img, None, fx=scale, fy=scale)
            if scale < 1:
                # Pad to original size
                pad_h = (h - scaled.shape[0]) // 2
                pad_w = (w - scaled.shape[1]) // 2
                padded = cv2.copyMakeBorder(scaled, pad_h, h - scaled.shape[0] - pad_h,
                                           pad_w, w - scaled.shape[1] - pad_w, cv2.BORDER_REFLECT)
                augmented.append(padded)
            else:
                # Center crop to original size
                start_h = (scaled.shape[0] - h) // 2
                start_w = (scaled.shape[1] - w) // 2
                cropped = scaled[start_h:start_h+h, start_w:start_w+w]
                augmented.append(cropped)
        
        # Flip + brightness combo
        flipped = cv2.flip(img, 1)
        bright_flip = np.clip(flipped.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        augmented.append(bright_flip)
    
    return augmented


def load_images_from_folder(folder_path, label, augment=True):
    """Load all images from a folder and extract features with data augmentation."""
    X, y = [], []
    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder not found: {folder_path}")
        return X, y
    
    files = [f for f in os.listdir(folder_path) 
             if os.path.splitext(f.lower())[1] in supported_ext]
    
    print(f"  📁 Found {len(files)} images in {folder_path}")
    
    for i, filename in enumerate(files):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.resize(img, (224, 224))
        
        if augment:
            # Create augmented versions
            augmented_images = augment_image(img)
            for aug_img in augmented_images:
                features = extract_features_from_array(aug_img)
                X.append(features)
                y.append(label)
        else:
            features = extract_features_from_array(img)
            X.append(features)
            y.append(label)
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(files)} images...")
    
    if augment:
        print(f"  📈 Augmented to {len(X)} samples")
    
    return X, y
    
    return X, y


def load_training_data():
    """Load training data from your dataset folders."""
    print("🔄 Loading your training data...")
    X, y = [], []
    
    # Define training data folders
    safe_folder = os.path.join(os.path.dirname(__file__), "training_data", "safe")
    unsafe_folder = os.path.join(os.path.dirname(__file__), "training_data", "unsafe")
    
    # Load real images from folders
    print("\n📂 Loading SAFE images...")
    safe_X, safe_y = load_images_from_folder(safe_folder, 0)  # 0 = safe
    X.extend(safe_X)
    y.extend(safe_y)
    
    print("\n📂 Loading UNSAFE images...")
    unsafe_X, unsafe_y = load_images_from_folder(unsafe_folder, 1)  # 1 = unsafe
    X.extend(unsafe_X)
    y.extend(unsafe_y)
    
    # Require real images - no synthetic fallback
    if len(X) == 0:
        print("\n❌ ERROR: No images found!")
        print("   Please add your images to:")
        print(f"   - {safe_folder}  (for safe/kid-friendly images)")
        print(f"   - {unsafe_folder}  (for unsafe images)")
        print("\n   Supported formats: JPG, JPEG, PNG, BMP, GIF, WEBP")
        return None, None
    
    if len(safe_X) == 0:
        print(f"\n❌ ERROR: No SAFE images found in {safe_folder}")
        return None, None
    
    if len(unsafe_X) == 0:
        print(f"\n❌ ERROR: No UNSAFE images found in {unsafe_folder}")
        return None, None
    
    print(f"\n✅ Loaded {len(safe_X)} safe + {len(unsafe_X)} unsafe = {len(X)} total images")
    
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────

def train_and_save_model():
    print("=" * 60)
    print("🚀 Kids Safety Image Classifier - Model Training")
    print("=" * 60)

    X, y = load_training_data()
    
    # Check if data was loaded successfully
    if X is None or y is None:
        print("\n❌ Training aborted: No training data available.")
        print("   Add your images to the training_data folders and try again.")
        return None
    
    print(f"\n📊 Dataset shape: {X.shape}, Labels: {np.bincount(y)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🏋️  Training ensemble model...")
    
    # Calculate class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"📊 Class weights: Safe={class_weights[0]:.2f}, Unsafe={class_weights[1]:.2f}")

    # Enhanced ensemble with more estimators and tuned hyperparameters
    from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    
    rf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=25, 
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42, 
        n_jobs=-1
    )
    
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=300, 
        max_depth=8, 
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        random_state=42
    )
    
    svm = SVC(
        kernel='rbf', 
        probability=True, 
        C=50, 
        gamma='scale', 
        class_weight='balanced',
        random_state=42
    )
    
    ada = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )
    
    # Stacking ensemble with logistic regression meta-learner
    base_estimators = [
        ('rf', rf),
        ('et', et),
        ('gb', gb),
        ('svm', svm),
        ('ada', ada)
    ]
    
    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1, max_iter=1000, class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print("   Training base models...")
    ensemble.fit(X_train_scaled, y_train)
    
    # Calibrate probabilities for better confidence estimates
    print("   Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated.fit(X_train_scaled, y_train)
    
    y_pred = calibrated.predict(X_test_scaled)
    y_proba = calibrated.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))
    
    # Detailed metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    print(f"\n📊 Detailed Metrics:")
    print(f"   F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")

    # Cross-validation on base ensemble
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\n📊 Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(calibrated, f)  # Save calibrated model
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata
    metadata = {
        "accuracy": float(acc),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feature_dim": int(X.shape[1]),
        "classes": ["Safe", "Unsafe"],
        "model_type": "Stacking Ensemble (RF + ET + GB + SVM + AdaBoost) + Calibration",
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba[:, 1]))
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Model saved to {model_dir}/")
    print("=" * 60)
    return acc


if __name__ == "__main__":
    train_and_save_model()
