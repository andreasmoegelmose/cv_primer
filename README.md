# Computer Vision Primer: Exercise Handout

Two short Python exercises, one at the end of each section.

**Setup (do this before the session):**
```
pip install opencv-python scikit-image numpy matplotlib
```
Everything also runs in Google Colab with the same line.

| Exercise | Time | Topic |
|---|---|---|
| 1 | 10 min | Segment and count objects |
| 2 | 15 min | Build a Harris corner detector |

Lines marked `TODO` are yours to write. Do the numbered tasks in order and note down what you observe. Faster than expected? Do the stretch task.

---

## Exercise 1 (10 min): Count the coins

**Goal:** Segment objects with a threshold, clean the mask with a median filter, and count them with connected components.

```python
import cv2, numpy as np
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
```

**Tasks**
1. Get the count as close as possible to the true number of coins (24). Look at the Otsu result first: does anything look wrong?
2. Filter out tiny components by area. Why does this help?
3. *Stretch:* use `cv2.Canny` instead and compare. When is edge-based better than threshold-based?

**Think about:** What assumption does Otsu make about the image, and when does it fail?

*Hints (try without them first):* `cv2.medianBlur(mask, 5)`. If one component is far larger than the others, look at where the background is brighter.

---

## Exercise 2 (15 min): Build a Harris corner detector

**Goal:** Build the Harris detector from the pieces in the lecture, using tools from Section 1 (Sobel gradients, Gaussian smoothing, connected components): structure tensor → eigenvalues → corner response. A checkerboard gives a known answer, and OpenCV's `cornerHarris` is a reference to check yourself against.

```python
import cv2, numpy as np
import matplotlib.pyplot as plt
from skimage import data


def structure_tensor(img, sigma=2):
    """Per-pixel structure tensor: Sobel gradients, products, Gaussian window."""
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    Sxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
    Syy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)
    return Sxx, Syy, Sxy


def eigenvalues(Sxx, Syy, Sxy):
    """Eigenvalues of [[Sxx, Sxy], [Sxy, Syy]] at every pixel, lambda1 >= lambda2."""
    half_trace = (Sxx + Syy) / 2
    root = np.sqrt(((Sxx - Syy) / 2) ** 2 + Sxy ** 2)
    return half_trace + root, half_trace - root


# ---------------------------------------------------------------------------
# Step 1: Flat, edge or corner?
# A checkerboard with 25 px squares. Look at the three marked points and
# predict lambda1 and lambda2 at each of them BEFORE you read the printout.
# ---------------------------------------------------------------------------
board = data.checkerboard().astype(np.float32) / 255
points = {"flat": (12, 12), "edge": (25, 12), "corner": (25, 25)}

l1, l2 = eigenvalues(*structure_tensor(board))
for name, (x, y) in points.items():
    print(f"{name:6s} lambda1 {l1[y, x]:6.3f}  lambda2 {l2[y, x]:6.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(board, cmap="gray", vmin=-0.5, vmax=1.5)
for name, (x, y) in points.items():
    ax[0].plot(x, y, "ro")
    ax[0].annotate(name, (x, y), xytext=(x + 4, y - 4), color="red", fontsize=12)
ax[0].set_xlim(0, 75); ax[0].set_ylim(75, 0)
ax[0].set_title("checkerboard (zoom on top-left corner)")
ax[1].imshow(l1, cmap="magma"); ax[1].set_title("lambda1 (larger eigenvalue)")
ax[2].imshow(l2, cmap="magma"); ax[2].set_title("lambda2 (smaller eigenvalue)")
for a in ax[1:]:
    a.set_xlim(0, 75); a.set_ylim(75, 0)
plt.show()

exit()  # TODO: Remove when you are ready for step 2


# ---------------------------------------------------------------------------
# Step 2: Harris corner response
# Complete R below. A correct R finds all 7 x 7 = 49 inner checkerboard corners.
# ---------------------------------------------------------------------------
def harris(img, k=0.05, sigma=2, rel_thresh=0.01):
    Sxx, Syy, Sxy = structure_tensor(img, sigma)
    R = np.zeros_like(Sxx)   # <- TODO: det - k * trace^2 of the structure tensor
    peaks = (R == cv2.dilate(R, np.ones((7, 7), np.uint8))) & (R > rel_thresh * R.max())
    # a corner between pixels gives several tied maxima: merge them (connected components)
    _, _, _, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
    return R, centroids[1:]                                # (x, y) per corner


R_board, pts_board = harris(board)
print("checkerboard corners:", len(pts_board), "(expected 49)")

img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
R, pts = harris(img)
R_cv = cv2.cornerHarris(img, blockSize=5, ksize=3, k=0.05)
print("astronaut corners:", len(pts))
if R.any():
    print("correlation with cv2.cornerHarris:", round(np.corrcoef(R.ravel(), R_cv.ravel())[0, 1], 2))

lim = max(abs(R_board).max(), 1e-9)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(board, cmap="gray", vmin=-0.5, vmax=1.5)
ax[0].plot(pts_board[:, 0], pts_board[:, 1], "ro", ms=6)
ax[0].set_title(f"checkerboard: {len(pts_board)} corners")
im = ax[1].imshow(R_board, cmap="coolwarm", vmin=-lim, vmax=lim)
ax[1].set_title("your R (red > 0, blue < 0)")
fig.colorbar(im, ax=ax[1], shrink=0.8)
ax[2].imshow(img, cmap="gray"); ax[2].plot(pts[:, 0], pts[:, 1], "r+")
ax[2].set_title(f"astronaut: {len(pts)} corners")
for a in ax:
    a.axis("off")
plt.show()

exit()  # TODO: Remove when you are ready for step 3


# ---------------------------------------------------------------------------
# Step 3: Window size and threshold
# sigma sets how large a neighbourhood counts as "the patch".
# Then try rel_thresh=0.001 and rel_thresh=0.1 in the harris() call below.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, sigma in zip(ax, [1, 2, 4]):
    p = harris(img, sigma=sigma)[1]
    a.imshow(img, cmap="gray"); a.plot(p[:, 0], p[:, 1], "r+")
    a.set_title(f"sigma {sigma}: {len(p)} corners"); a.axis("off")
plt.show()
```

**Tasks**
The script runs one step at a time: it stops at each `exit()  # TODO: Remove`. Delete that line when you are ready for the next step.

1. **Flat, edge or corner:** Run the script. Look at the three marked points on the checkerboard and predict λ1 and λ2 at each before you read the printout. Does it match the lecture? Where are the eigenvalue maps bright?
2. **Harris response:** Remove the first `exit()` and complete `R` in step 2. Do you get 49 checkerboard corners? Where is R positive, negative and zero? How close is your R to `cv2.cornerHarris`?
3. **Window size and threshold:** Remove the second `exit()`. Look at the plots for `sigma` 1, 2 and 4, then try `rel_thresh` 0.001 and 0.1. Which corners survive a large window? What does that say about Harris and image scale?
4. *Stretch:* try `k` = 0.04, 0.15 and 0.25. Why does everything disappear at 0.25? Or: compute and visualise HOG on the astronaut with `skimage.feature.hog(..., visualize=True)`.

**Think about:** Why does a corner need *two* large eigenvalues?

*Hints:* det = Sxx·Syy − Sxy², trace = Sxx + Syy. All three are images, so the formula works on whole arrays at once. At a perfect corner λ1 = λ2.
