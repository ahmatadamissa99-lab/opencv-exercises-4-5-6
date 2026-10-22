"""
Exercise 4: Detection of Lines, Circles and Human Faces
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def file_path(filename):
    return os.path.join(SCRIPT_DIR, filename)

rng = np.random.default_rng(3)

# ------------------------------------------------------------------
# Sample images (generated automatically if the files do not exist).
# You can replace them with your own photos:
#   shapes.jpg -> roads / buildings / geometric shapes
#   coins.jpg  -> coins or other circular objects
#   faces.jpg  -> a photo with one or more human faces
# ------------------------------------------------------------------
if not os.path.exists(file_path("shapes.jpg")):
    s = np.full((400, 600, 3), 240, dtype=np.uint8)
    cv2.rectangle(s, (60, 60), (260, 220), (0, 0, 0), 3)
    cv2.line(s, (60, 220), (260, 60), (0, 0, 0), 3)
    cv2.fillPoly(s, [np.array([[320, 330], [560, 330], [440, 160]])], (60, 60, 60))
    cv2.line(s, (30, 300), (300, 380), (0, 0, 0), 3)
    cv2.line(s, (350, 40), (580, 120), (0, 0, 0), 3)
    cv2.imwrite(file_path("shapes.jpg"), s)

if not os.path.exists(file_path("coins.jpg")):
    c = np.full((400, 600, 3), 200, dtype=np.uint8)
    for (cx, cy, r) in [(110, 110, 60), (260, 120, 45), (400, 100, 70),
                        (150, 290, 50), (300, 300, 65), (470, 290, 40)]:
        cv2.circle(c, (cx, cy), r, (70, 130, 190), -1)              # coin body
        cv2.circle(c, (cx, cy), r, (40, 90, 150), 3)                # coin rim
        cv2.circle(c, (cx, cy), int(r * 0.6), (90, 150, 210), 2)    # inner ring
    c = np.clip(c + rng.normal(0, 5, c.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(file_path("coins.jpg"), c)

if os.path.exists(file_path("faces.jpg")):
    faces_image_path = file_path("faces.jpg")
else:
    faces_image_path = file_path("astronaut.png")
    print("Using the bundled astronaut image as the face-detection sample.")

# =====================================================
#  Hough Line Transform
# =====================================================
img_lines = cv2.imread(file_path("shapes.jpg"))
if img_lines is None:
    raise FileNotFoundError("Could not load shapes.jpg.")
gray = cv2.cvtColor(img_lines, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Probabilistic Hough: returns line segments (x1, y1, x2, y2)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80,
                        minLineLength=60, maxLineGap=10)

result_lines = img_lines.copy()
n_lines = 0
if lines is not None:
    n_lines = len(lines)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(result_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
print("Lines detected:", n_lines)
cv2.putText(result_lines, f"Lines: {n_lines}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(file_path("lines_result.png"), result_lines)

# =====================================================
#   Hough Circle Transform
# =====================================================
img_circles = cv2.imread(file_path("coins.jpg"))
if img_circles is None:
    raise FileNotFoundError("Could not load coins.jpg.")
gray_c = cv2.cvtColor(img_circles, cv2.COLOR_BGR2GRAY)
gray_c = cv2.medianBlur(gray_c, 9)                       # strong blur: hides inner details, keeps the outer rim

circles = cv2.HoughCircles(gray_c, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,   # min distance between circle centres
                           param1=100,     # upper Canny threshold
                           param2=40,      # accumulator threshold (lower = more circles)
                           minRadius=35, maxRadius=90)

result_circles = img_circles.copy()
n_circles = 0
if circles is not None:
    circles = np.round(circles[0]).astype(int)
    n_circles = len(circles)
    for (x, y, r) in circles:
        cv2.circle(result_circles, (x, y), r, (0, 255, 0), 3)   # outline
        cv2.circle(result_circles, (x, y), 3, (0, 0, 255), -1)  # centre
print("Circles detected:", n_circles)
cv2.putText(result_circles, f"Circles: {n_circles}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(file_path("circles_result.png"), result_circles)

# =====================================================
#   Face detection with a pretrained Haar cascade
# =====================================================
img_faces = cv2.imread(faces_image_path)
n_faces = 0
if img_faces is None:
    raise FileNotFoundError(
        "Could not load the face sample. Add faces.jpg or restore astronaut.png.")

cascade_name = "haarcascade_frontalface_default.xml"
cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
if not hasattr(cv2, "CascadeClassifier"):
    raise RuntimeError(
        "This exercise requires OpenCV 4 with Haar cascade support. "
        "Install the packages in requirements.txt and use the project virtual environment.")

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    cascade_path = file_path(cascade_name)
    face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise FileNotFoundError(f"Could not load the Haar cascade: {cascade_path}")

gray_f = cv2.cvtColor(img_faces, cv2.COLOR_BGR2GRAY)
gray_f = cv2.equalizeHist(gray_f)                 # helps with uneven lighting
detected_faces = face_cascade.detectMultiScale(
    gray_f, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
n_faces = len(detected_faces)
for x, y, w, h in detected_faces:
    cv2.rectangle(img_faces, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.putText(img_faces, f"Faces: {n_faces}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite(file_path("faces_result.png"), img_faces)
print("Faces detected:", n_faces)
print("Face detector: Haar cascade")

# --- Optional: the same detector on a webcam / video (press q to quit) ---
# cap = cv2.VideoCapture(0)              # or cv2.VideoCapture("video.mp4")
# while True:
#     ok, frame = cap.read()
#     if not ok: break
#     g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     for (x, y, w, h) in face_cascade.detectMultiScale(g, 1.1, 5, minSize=(40, 40)):
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.imshow("Faces", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"): break
# cap.release(); cv2.destroyAllWindows()

# ---------------- Summary figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(12, 5))
manager = fig.canvas.manager
manager.set_window_title("Exercise 4 - Detection Results")
ax[0].imshow(cv2.cvtColor(result_lines, cv2.COLOR_BGR2RGB));   ax[0].set_title("4.a Hough lines")
ax[1].imshow(cv2.cvtColor(result_circles, cv2.COLOR_BGR2RGB)); ax[1].set_title("4.b Hough circles")
if img_faces is not None:
    ax[2].imshow(cv2.cvtColor(img_faces, cv2.COLOR_BGR2RGB))
ax[2].set_title("4.c Haar cascade faces")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(file_path("exercise4_results.png"), dpi=110)
manager.window.state("normal")
manager.window.lift()
plt.show(block=True)