"""Strip the dye out of each fabric's base colour, leaving only the weave.

The tint in the studio multiplies the base colour map. When that map already carries
the fabric's own dye, the two multiply together: asking for blue on ink velvet gives
you near-black, and asking for the same blue on ivory linen gives you something three
times brighter. The colour picker is then not really a colour picker, and two parts
"in the same colour" do not match.

So each fabric gets a second base colour map with its dye divided out, normalised so
the mean sits near white. Multiplying that by a tint gives the tint back, with the
weave's own light and shade riding on top, which is what a dye lot actually does to a
cloth.

The map is normalised to 0.85 rather than to 1.0 on purpose: a weave has highlights
above its own mean, and normalising to pure white would clip them flat in eight bits.

Run: python tools/neutralise.py
"""
import glob
import os

import numpy as np
from PIL import Image

TARGET_MEAN = 0.85
MATERIALS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "materials")


def to_linear(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def to_srgb(a):
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.clip(a, 0, None) ** (1 / 2.4) - 0.055)


for path in sorted(glob.glob(os.path.join(MATERIALS, "*_basecolor.jpg"))):
    slug = os.path.basename(path).split("_")[0]
    srgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    lin = to_linear(srgb)

    # The dye is the per-channel mean. Dividing it out in LINEAR space is the part
    # that matters: doing it in sRGB would leave a colour cast that gets worse the
    # further the fabric is from mid grey.
    dye = lin.mean(axis=(0, 1))
    native = to_srgb(dye)
    weave = lin / np.maximum(dye, 1e-6)

    # renormalise to the target mean and clip the few pixels that still overshoot
    weave *= TARGET_MEAN / max(weave.mean(), 1e-6)
    out = np.clip(to_srgb(weave), 0, 1)

    dest = os.path.join(MATERIALS, f"{slug}_weave.jpg")
    Image.fromarray((out * 255).astype(np.uint8)).save(dest, quality=92, optimize=True)

    hexcol = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in native)
    print(f"{slug:<14} native {hexcol}   weave mean {out.mean():.3f}   -> {os.path.basename(dest)}")
