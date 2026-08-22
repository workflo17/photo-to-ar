"""Emit the photograph's subject silhouette as a square bitmask, for alignment.

The matte already decides what is product and what is backdrop. This re-uses that
decision rather than re-deciding it, so the mask the alignment scores against is exactly
the mask the reconstruction was fed.

Written as raw bytes, one per pixel, so the Node side can read it without a PNG decoder.

Usage: python lib/photomask.py <matted.png> <out.bin> [size]
"""
import sys

import numpy as np
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 192

# Prefer the matte's own alpha. Reconstructing the silhouette from the composite is
# hopeless whenever the product is near the backdrop colour, which for a white product on
# a white seamless is always: the recovered mask comes back as the shadows.
import os

alpha_path = SRC.rsplit(".", 1)[0] + "-alpha.png"
if os.path.exists(alpha_path):
    subject = np.asarray(Image.open(alpha_path).convert("L")) > 127
else:
    rgb = np.asarray(Image.open(SRC).convert("RGB")).astype(np.float64)
    subject = np.linalg.norm(255.0 - rgb, axis=-1) > 18.0
    print("no alpha beside the matte, falling back to distance from white",
          file=sys.stderr)

ys, xs = np.nonzero(subject)
if not len(ys):
    print("no subject found in the matte", file=sys.stderr)
    sys.exit(1)

# Crop to the subject's bounding box and letterbox it into a square. The mesh
# silhouette is fitted the same way, so the two are compared on shape alone and neither
# framing nor absolute scale can affect the score.
top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
crop = subject[top:bottom + 1, left:right + 1]
h, w = crop.shape
span = max(h, w)
square = np.zeros((span, span), bool)
square[(span - h) // 2:(span - h) // 2 + h, (span - w) // 2:(span - w) // 2 + w] = crop

small = np.asarray(Image.fromarray((square * 255).astype(np.uint8)).resize(
    (SIZE, SIZE), Image.BILINEAR)) > 127

open(OUT, "wb").write(small.astype(np.uint8).tobytes())
print(f"mask {SIZE}x{SIZE}, subject {100.0 * small.mean():.1f}% of the square, "
      f"source box {right - left + 1}x{bottom - top + 1}")
