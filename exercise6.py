"""
Exercise 6: Image Segmentation Using K-means Clustering (OpenCV)
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILE = os.path.join(SCRIPT_DIR, "sample.jpg")  # replace with your own color photo
cv2.setRNGSeed(42)

# If no image is available, use the sample photo bundled with scikit-learn
if not os.path.exists(IMAGE_FILE):
    from sklearn.datasets import load_sample_image
    cv2.imwrite(IMAGE_FILE, cv2.cvtColor(load_sample_image("china.jpg"), cv2.COLOR_RGB2BGR))

# =====================================================
# 6.a  Load the image and reshape the pixels for clustering
# =====================================================
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError("Image not found: check the file path.")

# Resize very large images so that K-means stays fast
max_side = 800
scale = max_side / max(img.shape[:2])
if scale < 1:
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

h, w, ch = img.shape
pixels = img.reshape((-1, 3))              # (H, W, 3) -> (H*W, 3): one row per pixel (B, G, R)
pixels = np.float32(pixels)                # cv2.kmeans needs float32
print(f"Image: {w}x{h} pixels  ->  data matrix shape: {pixels.shape}")


# =====================================================
# 6.b  K-means segmentation
# =====================================================
def segment_kmeans(pixels, k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)                    # cluster colours (B, G, R)
    segmented = centers[labels.flatten()]          # replace every pixel by its cluster colour
    segmented = segmented.reshape((h, w, 3))       # back to the image shape
    return segmented, labels.flatten(), centers, compactness


seg4, labels4, centers4, comp4 = segment_kmeans(pixels, 4)
cv2.imwrite(os.path.join(SCRIPT_DIR, "segmented_k4.png"), seg4)

# =====================================================
# 6.d  Repeat with six clusters
# =====================================================
seg6, labels6, centers6, comp6 = segment_kmeans(pixels, 6)
cv2.imwrite(os.path.join(SCRIPT_DIR, "segmented_k6.png"), seg6)

for k, labels, centers, comp in [(4, labels4, centers4, comp4), (6, labels6, centers6, comp6)]:
    print(f"\nK = {k}  | compactness (sum of squared distances) = {comp:,.0f}"
          f"  | mean error per pixel = {np.sqrt(comp / len(pixels)):.1f}")
    for i, c in enumerate(centers):
        share = np.mean(labels == i) * 100
        print(f"  cluster {i}: BGR = {c.tolist()}  -> {share:5.1f} % of the pixels")
print("\nObservation: K = 6 represents the image with more color clusters than K = 4.")
print("The clusters represent color groups, not necessarily separate physical objects.")
if comp6 < comp4:
    print(f"Compactness decreases by {comp4 - comp6:,.0f} when increasing K from 4 to 6.")


# =====================================================
# 6.c  Display the original image next to the segmented images
# =====================================================
def palette_bar(centers, labels):
    """A colour bar whose widths are proportional to the share of each cluster."""
    bar = np.zeros((40, 600, 3), np.uint8)
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(-counts)
    boundaries = np.rint(np.cumsum(counts[order] / counts.sum() * bar.shape[1])).astype(int)
    start = 0
    for cluster, end in zip(order, boundaries):
        bar[:, start:end] = centers[cluster]
        start = end
    return bar


fig = plt.figure(figsize=(16, 7))
fig.canvas.manager.set_window_title("Exercise 6 - K-means Image Segmentation")
gs = fig.add_gridspec(2, 3, height_ratios=[10, 1])
ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
bars = [fig.add_subplot(gs[1, i]) for i in range(3)]

ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Original")
ax[1].imshow(cv2.cvtColor(seg4, cv2.COLOR_BGR2RGB)); ax[1].set_title("K-means, K = 4")
ax[2].imshow(cv2.cvtColor(seg6, cv2.COLOR_BGR2RGB)); ax[2].set_title("K-means, K = 6")
bars[1].imshow(cv2.cvtColor(palette_bar(centers4, labels4), cv2.COLOR_BGR2RGB))
bars[2].imshow(cv2.cvtColor(palette_bar(centers6, labels6), cv2.COLOR_BGR2RGB))
bars[0].axis("off")
for a in ax + bars:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "exercise6_results.png"), dpi=110)
plt.show(block=True)