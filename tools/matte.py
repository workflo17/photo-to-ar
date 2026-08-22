"""
Step 1b: cut the product off its backdrop.

Single-image reconstruction has no idea what is product and what is set. Feed it a
studio shot and it will happily reconstruct the seamless and the contact shadow as a
slab of geometry under the chair, which is exactly what happened on the first pass.

The matte is a flood fill from the frame edge in Lab, which is the right space for
"is this the same grey" on a seamless backdrop, followed by a small erode and a
feather so the silhouette does not come out with a hard jaggy edge. The subject is
then laid on white, because that is what the reconstruction model was trained on.
"""
import numpy as np
from PIL import Image, ImageFilter
from collections import deque
from pathlib import Path

PIPE = Path(__file__).resolve().parent.parent / "docs" / "pipeline"
SRC = PIPE / "01-source.png"
OUT = PIPE / "01-matted.png"
TOLERANCE = 15.0        # Lab distance counted as "still the backdrop"


def srgb_to_lab(rgb):
    a = rgb.astype(np.float64) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


img = Image.open(SRC).convert("RGB")
rgb = np.asarray(img)
h, w, _ = rgb.shape
lab = srgb_to_lab(rgb)

# the backdrop colour is whatever the frame edge is
edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
backdrop = np.median(edge, axis=0)

dist = np.linalg.norm(lab - backdrop, axis=-1)
candidate = dist < TOLERANCE

# Only fill what actually connects to the edge. A cream cushion can sit inside the
# tolerance too, and blanket-thresholding would punch holes straight through it.
bg = np.zeros((h, w), bool)
q = deque()
for x in range(w):
    for y in (0, h - 1):
        if candidate[y, x] and not bg[y, x]:
            bg[y, x] = True
            q.append((y, x))
for y in range(h):
    for x in (0, w - 1):
        if candidate[y, x] and not bg[y, x]:
            bg[y, x] = True
            q.append((y, x))

while q:
    y, x = q.popleft()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and candidate[ny, nx] and not bg[ny, nx]:
            bg[ny, nx] = True
            q.append((ny, nx))

alpha = Image.fromarray(((~bg) * 255).astype(np.uint8))
# erode by a pixel to drop the halo of blended backdrop, then feather the cut
alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))

subject = np.asarray(alpha, dtype=np.float64)[..., None] / 255.0
on_white = (rgb * subject + 255 * (1 - subject)).astype(np.uint8)
Image.fromarray(on_white).save(OUT)

kept = subject.mean()
ys, xs = np.where(np.asarray(alpha) > 128)
print(f"matte: {kept * 100:.1f}% of the frame kept")
print(f"subject bounds: x {xs.min()}-{xs.max()}  y {ys.min()}-{ys.max()}  of {w}x{h}")
print("->", OUT.name)
