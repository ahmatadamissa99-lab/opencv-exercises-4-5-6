"""
Exercise 5: Day / Night Image Classification Using Feature Engineering
"""
import os
import glob
import argparse
import cv2
import numpy as np
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
DEMO_DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_demo")
# Store real images in dataset/day and dataset/night; use --demo for synthetic scenes.
N_PER_CLASS = 30               # used only when the demo dataset is generated
CLASSES = ["day", "night"]     # label 0 = day, label 1 = night
rng = np.random.default_rng(42)

parser = argparse.ArgumentParser(
    description="Classify day and night images using engineered features.")
parser.add_argument(
    "--demo", action="store_true",
    help="Use generated synthetic scenes for demonstration only.")
parser.add_argument(
    "image", nargs="?",
    help="Optional image path to classify after training.")
args = parser.parse_args()

if args.demo:
    DATASET_DIR = DEMO_DATASET_DIR
    demo_mode = True
else:
    demo_mode = False


# =====================================================
# 5.a  Prepare a labelled dataset (>= 20 day + >= 20 night)
# =====================================================
def make_demo_image(kind):
    """Create a synthetic day or night scene (only used if you have no real photos)."""
    h, w = 240, 320
    img = np.zeros((h, w, 3), np.float32)
    horizon = int(rng.integers(120, 180))
    yy = np.linspace(0, 1, horizon)[:, None]

    if kind == "day":
        sky = float(rng.uniform(95, 235))                    # overcast ... bright sky
        tint = rng.uniform(-25, 25, 3)
        for c in range(3):
            img[:horizon, :, c] = np.clip(sky + tint[c] - 30 * yy, 0, 255)
        ground = float(rng.uniform(25, 120))                 # trees, roads, buildings
        img[horizon:] = ground + rng.uniform(-15, 15, 3)
        if rng.random() < 0.6:                               # sun
            cv2.circle(img, (int(rng.integers(30, 290)), int(rng.integers(25, 80))),
                       int(rng.integers(12, 25)), (200, 245, 255), -1)
    else:
        sky = float(rng.uniform(2, 55))                      # night ... twilight
        tint = rng.uniform(0, 18, 3)
        for c in range(3):
            img[:horizon, :, c] = np.clip(sky + tint[c] + 12 * yy, 0, 255)
        img[horizon:] = rng.uniform(1, 28)
        for _ in range(int(rng.integers(0, 60))):            # stars
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, horizon))), 1,
                       (230, 230, 230), -1)
        if rng.random() < 0.5:                               # moon
            cv2.circle(img, (int(rng.integers(30, 290)), int(rng.integers(25, 80))),
                       int(rng.integers(8, 18)), (225, 235, 240), -1)
        for _ in range(int(rng.integers(0, 12))):            # street lights / windows
            x, y = int(rng.integers(0, w)), int(rng.integers(horizon - 30, h))
            cv2.circle(img, (x, y), int(rng.integers(3, 8)), (120, 220, 255), -1)

    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 2.0)))
    img += rng.normal(0, rng.uniform(2, 9), img.shape)       # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


if demo_mode:
    print("DEMO MODE: generated scenes are synthetic and are not a real-image evaluation.")
    for cls in CLASSES:
        folder = os.path.join(DATASET_DIR, cls)
        os.makedirs(folder, exist_ok=True)
        if len(glob.glob(os.path.join(folder, "*.*"))) == 0:
            print(f"No images in '{folder}': generating {N_PER_CLASS} demo images.")
            for i in range(N_PER_CLASS):
                cv2.imwrite(os.path.join(folder, f"{cls}_{i:02d}.jpg"), make_demo_image(cls))
elif not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Real dataset folder not found: '{DATASET_DIR}'. "
        "Create dataset/day and dataset/night and add at least 20 real images to each.")


# =====================================================
# 5.b  Feature extraction (3 features per image)
# =====================================================
def extract_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()                  # 1) average brightness
    std_intensity = gray.std()                     # 2) intensity standard deviation
    dark_ratio = np.mean(gray < 50)                # 3) proportion of dark pixels (< 50)
    return [mean_brightness, std_intensity, dark_ratio]


X, y, paths = [], [], []
for label, cls in enumerate(CLASSES):
    for p in sorted(glob.glob(os.path.join(DATASET_DIR, cls, "*.*"))):
        im = cv2.imread(p)
        if im is None:
            continue                               # skip files that are not images
        X.append(extract_features(im)); y.append(label); paths.append(p)

X, y, paths = np.array(X), np.array(y), np.array(paths)
print(f"Dataset: {len(y)} images  (day = {np.sum(y == 0)}, night = {np.sum(y == 1)})")
print("Dataset type:", "synthetic demonstration" if demo_mode else "user-provided images")
if np.sum(y == 0) < 20 or np.sum(y == 1) < 20:
    raise ValueError(
        f"At least 20 readable images per class are required in "
        f"'{DATASET_DIR}\\day' and '{DATASET_DIR}\\night'.")

# =====================================================
# 5.c  Train the classifiers (70 % train / 30 % held-out test)
# =====================================================
X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
    X, y, paths, test_size=0.30, stratify=y, random_state=42)

models = {
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# =====================================================
# 5.d  Evaluate on unseen images
# =====================================================
best_name, best_cv_score, best_model = None, -1.0, None
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_score = cv_scores.mean()
    print(f"\n{name}: 5-fold CV accuracy (training data) = "
          f"{cv_score:.3f} +/- {cv_scores.std():.3f}")
    if cv_score > best_cv_score:
        best_name, best_cv_score, best_model = name, cv_score, model

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nSelected model (by training CV): {best_name}")
print(f"Held-out test accuracy (reported once): {test_accuracy * 100:.1f} %")
print("\nConfusion matrix (rows = true, columns = predicted) [day, night]:")
print(confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))
joblib.dump(best_model, os.path.join(SCRIPT_DIR, "day_night_model.joblib"))

# ---------------- Figures ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.canvas.manager.set_window_title("Exercise 5 - Feature Space")
colors = ["gold" if l == 0 else "navy" for l in y]
ax[0].scatter(X[:, 0], X[:, 2], c=colors, edgecolors="k", alpha=0.8)
ax[0].set_xlabel("Average brightness"); ax[0].set_ylabel("Proportion of dark pixels")
ax[0].set_title("Feature space (yellow = day, blue = night)")
ax[1].scatter(X[:, 0], X[:, 1], c=colors, edgecolors="k", alpha=0.8)
ax[1].set_xlabel("Average brightness"); ax[1].set_ylabel("Intensity standard deviation")
ax[1].set_title("Feature space (yellow = day, blue = night)")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "exercise5_features.png"), dpi=110)

n_show = min(10, len(p_test))
fig, ax = plt.subplots(2, 5, figsize=(15, 6.5))
fig.canvas.manager.set_window_title("Exercise 5 - Test Predictions")
for a, p, t, pr in zip(ax.ravel(), p_test[:n_show], y_test[:n_show], y_pred[:n_show]):
    a.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    a.set_title(f"true: {CLASSES[t]} | pred: {CLASSES[pr]}",
                color="green" if t == pr else "red", fontsize=10)
    a.axis("off")
for a in ax.ravel()[n_show:]:
    a.axis("off")
plt.suptitle(f"Held-out test images - accuracy on the full test set: {test_accuracy * 100:.1f} %")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "exercise5_test_predictions.png"), dpi=110)


# --- Classify any new image:  python exercise5.py [--demo] my_photo.jpg ---
if args.image:
    image_path = args.image
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    else:
        pred = best_model.predict([extract_features(im)])[0]
        print(f"{image_path}  ->  {CLASSES[pred].upper()}")

plt.show(block=True)