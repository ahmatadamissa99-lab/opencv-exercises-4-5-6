"""
Exercise 3: Image Gradients, Thresholding, Edge Detection and Contours
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILE = os.path.join(SCRIPT_DIR, "objects.jpg")

# ------------------------------------------------------------------
# Sample image: if "objects.jpg" does not exist, one is generated
# (several coloured objects on a light background).
# You can replace it with your own photo (coins, fruits, bottles...).
# ------------------------------------------------------------------
if not os.path.exists(IMAGE_FILE):
    rng = np.random.default_rng(1)
    canvas = np.full((400, 600, 3), 225, dtype=np.uint8)           # light background
    cv2.circle(canvas, (100, 100), 50, (200, 60, 40), -1)
    cv2.rectangle(canvas, (200, 60), (310, 150), (40, 140, 40), -1)
    cv2.circle(canvas, (420, 110), 45, (30, 30, 200), -1)
    cv2.ellipse(canvas, (530, 100), (45, 30), 30, 0, 360, (20, 160, 220), -1)
    cv2.fillPoly(canvas, [np.array([[80, 330], [160, 330], [120, 240]])], (150, 40, 150))
    cv2.rectangle(canvas, (230, 250), (350, 340), (60, 90, 120), -1)
    cv2.circle(canvas, (450, 290), 40, (90, 90, 90), -1)
    noise = rng.normal(0, 6, canvas.shape)                         # a little noise
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(IMAGE_FILE, canvas)

img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError("Image not found: check the file path.")

# =====================================================
# 3.a  Grayscale conversion + binary thresholding
# =====================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)                # reduce noise first

# Otsu finds the best threshold automatically.
# THRESH_BINARY_INV: dark objects on a light background become WHITE (255).
# (If your objects are LIGHTER than the background, use cv2.THRESH_BINARY instead.)
thresh_value, binary = cv2.threshold(blur, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print("Otsu threshold value:", thresh_value)

# Light but coloured objects (e.g. yellow) can look like the background in
# grayscale. We also threshold the SATURATION channel and merge both masks.
saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
_, binary_sat = cv2.threshold(cv2.GaussianBlur(saturation, (5, 5), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary = cv2.bitwise_or(binary, binary_sat)

# Clean small holes / specks
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# =====================================================
# 3.b  Sobel gradients + Canny edge detection
# =====================================================
sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)    # horizontal gradient (vertical edges)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)    # vertical gradient (horizontal edges)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)
sobel_mag = cv2.convertScaleAbs(sobel_mag)              # back to 8-bit for display

canny = cv2.Canny(blur, 50, 150)                        # low / high hysteresis thresholds

# =====================================================
# 3.c  Contours
# =====================================================
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Ignore tiny contours (noise): keep only those with a minimum area
MIN_AREA = 500
objects = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

result = img.copy()
cv2.drawContours(result, objects, -1, (0, 255, 0), 3)

# =====================================================
# 3.d  Count the objects and display the number
# =====================================================
count = len(objects)
for i, c in enumerate(objects, start=1):
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        cv2.putText(result, str(i), (cx - 8, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

cv2.putText(result, f"Objects detected: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
print("Total number of objects detected:", count)

# ---------------- Save and display ----------------
cv2.imwrite(os.path.join(SCRIPT_DIR, "binary.png"), binary)
cv2.imwrite(os.path.join(SCRIPT_DIR, "sobel.png"), sobel_mag)
cv2.imwrite(os.path.join(SCRIPT_DIR, "canny.png"), canny)
cv2.imwrite(os.path.join(SCRIPT_DIR, "contours_result.png"), result)

fig, ax = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
fig.canvas.manager.set_window_title("Exercise 3 - Image Processing Results")
ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0, 0].set_title("Original")
ax[0, 1].imshow(gray, cmap="gray");                     ax[0, 1].set_title("Grayscale")
ax[0, 2].imshow(binary, cmap="gray");                   ax[0, 2].set_title("3.a Binary (Otsu)")
ax[1, 0].imshow(sobel_mag, cmap="gray");                ax[1, 0].set_title("3.b Sobel magnitude")
ax[1, 1].imshow(canny, cmap="gray");                    ax[1, 1].set_title("3.b Canny edges")
ax[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
ax[1, 2].set_title(f"3.c / 3.d Contours - {count} objects")
for a in ax.ravel():
    a.axis("off")
plt.savefig(os.path.join(SCRIPT_DIR, "exercise3_results.png"), dpi=120)
plt.show()