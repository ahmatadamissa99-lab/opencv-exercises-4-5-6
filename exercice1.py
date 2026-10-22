"""Exercise 1: Load, display, save, and draw images with OpenCV."""
import cv2
import numpy as np
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "image_source.jpg")

# ------------------------------------------------------------------
# Generate a sample image if 'image_source.jpg' does not exist.
# Replace it with any color photo from your computer if preferred.
# ------------------------------------------------------------------
if not os.path.exists(IMAGE_PATH):
    h, w = 400, 600
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    sample = np.zeros((h, w, 3), dtype=np.uint8)
    sample[:, :, 0] = x[None, :]            # Blue: horizontal gradient
    sample[:, :, 1] = y[:, None]            # Green: vertical gradient
    sample[:, :, 2] = 255 - x[None, :]      # Red: reversed gradient
    cv2.circle(sample, (150, 200), 70, (0, 255, 255), -1)
    cv2.rectangle(sample, (350, 100), (520, 300), (255, 255, 255), -1)
    cv2.imwrite(IMAGE_PATH, sample)

# ===================== QUESTION 1.a =====================
# Load a color image.
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")
print("Image dimensions (height, width, channels):", image.shape)

# Save a copy under a different name.
copy_path = os.path.join(SCRIPT_DIR, "image_copy.jpg")
cv2.imwrite(copy_path, image)
print("Image copy saved:", copy_path)

# Draw on a copy of the original photo, following the supplied example.
annotated_image = image.copy()
height, width = annotated_image.shape[:2]
scale = width / 700
line_thickness = max(2, round(6 * scale))

cv2.rectangle(
    annotated_image,
    (round(47 * scale), round(36 * scale)),
    (round(288 * scale), round(228 * scale)),
    (255, 0, 0),
    line_thickness,
)
cv2.circle(
    annotated_image,
    (round(447 * scale), round(138 * scale)),
    round(77 * scale),
    (0, 255, 0),
    line_thickness,
)
cv2.line(
    annotated_image,
    (round(48 * scale), round(312 * scale)),
    (round(600 * scale), round(312 * scale)),
    (0, 0, 255),
    line_thickness,
)
cv2.putText(
    annotated_image,
    "OpenCV",
    (round(62 * scale), round(405 * scale)),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.35 * scale,
    (255, 255, 0),
    max(2, round(3 * scale)),
    cv2.LINE_AA,
)
annotated_path = os.path.join(SCRIPT_DIR, "image_with_drawings.jpg")
cv2.imwrite(annotated_path, annotated_image)
print("Annotated photo saved:", annotated_path)

# ===================== QUESTION 1.b =====================
# Create a blank 500x500 image with a white background.
blank = np.ones((500, 500, 3), dtype=np.uint8) * 255
print("Blank white image created:", blank.shape)

# ===================== QUESTION 1.c =====================
# Draw shapes using BGR colors.
canvas = blank.copy()
cv2.rectangle(canvas, (50, 50), (200, 150), (255, 0, 0), 3)        # Blue rectangle
cv2.circle(canvas, (350, 120), 60, (0, 0, 255), -1)                # Filled red circle
cv2.line(canvas, (50, 250), (450, 250), (0, 150, 0), 4)            # Green line
cv2.putText(canvas, "OpenCV 2026", (90, 350),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)  # Black text

drawing_path = os.path.join(SCRIPT_DIR, "drawing_result.png")
cv2.imwrite(drawing_path, canvas)
print("Drawing saved:", drawing_path)

# Summary figure for the report.
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
fig.canvas.manager.set_window_title("Exercise 1 - Results")
ax.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
ax.axis("off")
summary_path = os.path.join(SCRIPT_DIR, "exercise1_summary.png")
plt.savefig(summary_path, dpi=120)
print("Summary saved:", summary_path)
plt.show(block=True)
