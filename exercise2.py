"""
Exercise 2: Image Blurring and Histogram Analysis
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILE = os.path.join(SCRIPT_DIR, "sample.jpg")

# Load a colour image (replace "sample.jpg" with your own picture)
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError("Image not found: check the file path.")
h, w = img.shape[:2]

# =====================================================
# 2.a  Gaussian blur on a selected rectangular region
# =====================================================
x1, y1, x2, y2 = 200, 100, 450, 300          # region of interest (ROI)

blurred = img.copy()
roi = blurred[y1:y2, x1:x2]                   # select the region
roi_blur = cv2.GaussianBlur(roi, (31, 31), 0) # (kernel size must be odd), sigma computed automatically
blurred[y1:y2, x1:x2] = roi_blur              # put the blurred region back

cv2.imwrite(os.path.join(SCRIPT_DIR, "blurred_region.jpg"), blurred)

# =====================================================
# 2.b  Grayscale histogram of the original image
# =====================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])

# =====================================================
# 2.c  Histogram equalisation and comparison
# =====================================================
equalized = cv2.equalizeHist(gray)
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

cv2.imwrite(os.path.join(SCRIPT_DIR, "gray.jpg"), gray)
cv2.imwrite(os.path.join(SCRIPT_DIR, "equalized.jpg"), equalized)

# ---------------- Display results ----------------
fig, ax = plt.subplots(2, 3, figsize=(11, 6.5))
manager = fig.canvas.manager
manager.set_window_title("Exercise 2 - Image Processing Results")

ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0, 0].set_title("Original image")
ax[0, 1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
ax[0, 1].set_title("2.a Gaussian blur on ROI only")
ax[0, 2].axis("off")

ax[1, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
ax[1, 0].set_title("Grayscale (original)")
ax[1, 1].imshow(equalized, cmap="gray", vmin=0, vmax=255)
ax[1, 1].set_title("2.c Equalised image")

ax[1, 2].plot(hist_original, color="black", label="Original")
ax[1, 2].plot(hist_equalized, color="red", label="Equalised")
ax[1, 2].set_title("2.b / 2.c Histograms")
ax[1, 2].set_xlabel("Pixel intensity"); ax[1, 2].set_ylabel("Number of pixels")
ax[1, 2].set_xlim([0, 255]); ax[1, 2].legend()

for a in ax[:2, :2].ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "exercise2_results.png"), dpi=120)
manager.window.state("normal")
manager.window.lift()
plt.show(block=True)

# Quick numeric comparison for your observations
print("Original  -> mean: %.1f | std: %.1f | min: %d | max: %d" %
      (gray.mean(), gray.std(), gray.min(), gray.max()))
print("Equalised -> mean: %.1f | std: %.1f | min: %d | max: %d" %
      (equalized.mean(), equalized.std(), equalized.min(), equalized.max()))
