"""Cut the product off its backdrop, before the model ever sees it.

Single-image reconstruction has no idea what is product and what is set. Feed it a
studio shot and it will happily reconstruct the seamless and the contact shadow as a
slab of geometry under the piece. Removing the backdrop first is strictly better than
cutting the slab off afterwards: geometry that was never generated costs nothing to
remove and never distorts the scale measurement.

The matte is a flood fill from the frame edge in Lab, which is the right space for
"is this the same grey" on a seamless backdrop, followed by a small erode and a
feather so the silhouette does not come out with a hard jaggy edge. The subject is
then laid on white, because that is what the reconstruction model was trained on.

Ported from ~/product-viz/tools/matte.py and parameterised. Reads and writes paths
given on the command line rather than a hardcoded pipeline directory.

Usage: python lib/matte.py <in.png> <out.png> [tolerance]
"""
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

SRC, OUT = sys.argv[1], sys.argv[2]
TOLERANCE = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0


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

# The backdrop colour is whatever the frame edge is.
edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
backdrop = np.median(edge, axis=0)
dist = np.linalg.norm(lab - backdrop, axis=-1)


def flood(tol):
    """Everything reachable from the frame edge within `tol` of the backdrop colour."""
    candidate = dist < tol
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
    return ~bg


# A fixed tolerance cannot work. It has to be loose enough to swallow a lit seamless and
# tight enough not to leak into the product, and where that line sits depends entirely on
# how close the two colours are. A cream chair on a light grey backdrop is about 8 Lab
# units apart, and the old fixed 15 walked straight through the chair, keeping 17% of the
# frame where the chair is about half of it.
#
# So sweep, and let the mask choose. The tell is solidity: the share of the subject that
# is in its single largest piece. While the flood stops at the real edge the subject stays
# in one piece and solidity is 1.0; the moment the tolerance leaks into the product the
# leak eats corridors through it and solidity falls off a cliff. Measured on the reference
# chair: 1.000 at tolerance 8 and 9, 0.878 at 10, 0.671 at 12.
#
# Take the LARGEST tolerance that is still solid, because that is the one that removes the
# most backdrop without touching the product. Picking the smallest instead keeps the chair
# intact and most of the seamless with it, which is safe and useless.
try:
    from scipy import ndimage

    chosen = None
    for tol in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18):
        mask = ndimage.binary_fill_holes(flood(tol))
        mask = ndimage.binary_closing(mask, np.ones((3, 3)), iterations=2)
        mask = ndimage.binary_fill_holes(mask)
        coverage = float(mask.mean())
        if coverage > 0.92 or coverage < 0.02:
            continue                      # found no backdrop, or ate the product
        labels, n = ndimage.label(mask)
        if not n:
            continue
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        solidity = float(sizes.max()) / max(float(mask.sum()), 1.0)
        if solidity >= 0.99:
            chosen = (tol, mask, coverage, solidity)
    if chosen is None:
        raise RuntimeError("no tolerance kept the subject in one piece")
    TOLERANCE, subject_mask, coverage, solidity = chosen
    print(f"tolerance {TOLERANCE}: {coverage:.1%} of the frame, solidity {solidity:.3f}")
except ImportError:
    print("scipy missing, using a fixed tolerance")
    subject_mask = flood(TOLERANCE)

alpha = Image.fromarray((subject_mask * 255).astype(np.uint8))
# erode by a pixel to drop the halo of blended backdrop, then feather the cut
alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))

subject = np.asarray(alpha, dtype=np.float64)[..., None] / 255.0
kept = float(subject.mean())

# A matte that kept everything found no backdrop; one that kept almost nothing ate the
# product. Either way the caller should use the original rather than trust this.
if kept > 0.985 or kept < 0.02:
    print(f"no usable backdrop found (kept {kept:.1%}), leaving the photo alone")
    sys.exit(2)

Image.fromarray((rgb * subject + 255 * (1 - subject)).astype(np.uint8)).save(OUT)

# Save the alpha too. Compositing onto white throws away the one thing that says where
# the product is, and a white product on white is then indistinguishable from the
# backdrop: the texture alignment tried to recover the silhouette from the composite and
# got the chair's shadows instead of the chair.
alpha_path = OUT.rsplit(".", 1)[0] + "-alpha.png"
Image.fromarray((np.asarray(alpha)).astype(np.uint8)).save(alpha_path)

print(f"matted: subject is {kept:.1%} of the frame")
