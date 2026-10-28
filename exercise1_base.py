import cv2
import matplotlib.pyplot as plt
from skimage import data

img = data.coins()                                  # uint8 grayscale

# 1. Denoise, then threshold (Otsu)
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2. TODO: clean up the mask with a median filter (try kernel sizes 3, 5, 7)
clean = bw  # <- replace

# 3. Count connected components (label 0 is background)
n, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
areas = stats[1:, cv2.CC_STAT_AREA]
print("components:", n - 1, "areas:", sorted(areas))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(ax, [img, bw, clean], ["input", "Otsu", "cleaned"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.show()