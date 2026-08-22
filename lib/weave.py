"""Generate a fabric texture set from parameters, on demand.

Builds a physically-motivated height field for the weave structure, then derives the
glTF texture set from it:

  <slug>_weave.jpg     the weave with no dye in it, mean near white, ready to tint
  <slug>_orm.jpg       packed R=occlusion  G=roughness  B=metalness  (glTF convention)
  <slug>_normal.png    tangent-space normal, OpenGL +Y-up (what glTF expects)

Weave model: a cell grid P[i,j] where 1 = warp yarn on top. Yarn crimp comes from
interpolating that pattern along the yarn's own length, so plain / twill / satin all
fall out of the same code. Boucle and velvet are pile structures, not interlacings, so
they get their own height fields.

Ported from ~/product-viz/tools/make_fabrics.py, which generated a fixed table of four
fabrics at build time. Here the parameters arrive on the command line, so the studio can
make a cloth nobody has made before. The base colour is written WITHOUT dye, because the
studio supplies colour as a tint: baking a colour in is what made every colourway come
out muddy.

Usage: python lib/weave.py <outdir> <slug> <weave> <roughness> <normalStrength> <threadCount>
"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path

SIZE = 1024
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SLUG = sys.argv[2]
WEAVE = sys.argv[3]
ROUGH = float(sys.argv[4])
NSTR = float(sys.argv[5])
THREADS = int(sys.argv[6])
# fixed seed: asking for the same cloth twice must give the same cloth
rng = np.random.default_rng(7)


def cos_lerp(a, b, t):
    f = (1.0 - np.cos(t * np.pi)) * 0.5
    return a * (1 - f) + b * f


def weave_pattern(kind, n):
    """Cell grid: 1 where the warp (vertical) yarn floats over the weft."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    if kind == "plain":                       # over 1, under 1
        return ((i + j) % 2 == 0).astype(np.float64)
    if kind == "twill":                       # 2/2 twill, diagonal wale
        return (((i - j) % 4) < 2).astype(np.float64)
    raise ValueError(kind)


def interlaced_height(kind, n, crimp=0.42, yarn=0.58, hairiness=0.05):
    """Height field for a real interlacing. Yarn cross-section is a semi-ellipse,
    crimp is the over/under undulation read off the weave pattern."""
    P = weave_pattern(kind, n)
    pitch = SIZE / n
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)

    ix = np.floor(x / pitch).astype(int) % n          # which warp yarn
    jy = np.floor(y / pitch).astype(int) % n          # which weft yarn
    u = (x / pitch) - ix - 0.5                        # -0.5..0.5 across warp width
    v = (y / pitch) - jy - 0.5

    # semi-elliptical yarn cross-sections
    hw = np.sqrt(np.clip(1.0 - (u / 0.5) ** 2, 0, 1)) * yarn
    hf = np.sqrt(np.clip(1.0 - (v / 0.5) ** 2, 0, 1)) * yarn

    # crimp: sample the pattern along the yarn's length and smooth it
    t_w = (y / pitch) - 0.5                           # warp runs along y
    j0 = np.floor(t_w).astype(int)
    fw = t_w - j0
    cw = cos_lerp(P[ix, j0 % n], P[ix, (j0 + 1) % n], fw)

    t_f = (x / pitch) - 0.5                           # weft runs along x
    i0 = np.floor(t_f).astype(int)
    ff = t_f - i0
    cf = cos_lerp(P[i0 % n, jy], P[(i0 + 1) % n, jy], ff)

    warp = (cw * 2 - 1) * crimp + hw
    weft = ((1 - cf) * 2 - 1) * crimp + hf
    h = np.maximum(warp, weft)

    # Loose fibre fuzz on the yarn crowns. The noise is blurred BEFORE it joins the
    # height, because the normal map is this field's one-pixel derivative and the
    # derivative of uncorrelated noise is uncorrelated noise: unblurred, the normal
    # map came out as full-amplitude per-pixel static that weighed 2.5 MB as a PNG
    # and lit as sandpaper rather than as fibre. Correlating the fuzz over a few
    # texels keeps the fibre look and gives the derivative something smooth to see.
    fuzz = rng.normal(0, hairiness, (SIZE, SIZE))
    k = np.array([1, 2, 1], dtype=np.float64); k /= k.sum()
    for ax in (0, 1):
        fuzz = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), ax, fuzz)
    h += fuzz * np.clip(h, 0, 1)
    return h, ix, jy


def boucle_height(n_loops=2600):
    """Bouclé: no regular interlacing, a ground weave buried under looped slubs."""
    h = np.zeros((SIZE, SIZE))
    base, _, _ = interlaced_height("plain", 70, crimp=0.18, yarn=0.30, hairiness=0.03)
    h += base * 0.35
    yy, xx = np.mgrid[-24:25, -24:25].astype(np.float64)
    r = np.hypot(xx, yy)
    for _ in range(n_loops):
        R = rng.uniform(6, 15)
        w = rng.uniform(1.6, 3.0)
        bump = np.exp(-((r - R) ** 2) / (2 * w * w)) * rng.uniform(0.6, 1.0)
        cy, cx = rng.integers(0, SIZE, 2)
        ys = (np.arange(-24, 25) + cy) % SIZE
        xs = (np.arange(-24, 25) + cx) % SIZE
        h[np.ix_(ys, xs)] = np.maximum(h[np.ix_(ys, xs)], bump)
    return h


def velvet_height():
    """Cut pile: the surface is flat, the light response comes from fibre ends."""
    fine = rng.normal(0, 1, (SIZE, SIZE))
    k = np.array([1, 4, 6, 4, 1], dtype=np.float64); k /= k.sum()
    for ax in (0, 1):                                  # cheap separable blur
        fine = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), ax, fine)
    nap = np.sin(np.mgrid[0:SIZE, 0:SIZE][0] * 0.09) * 0.015  # faint pressed nap
    return fine * 0.10 + nap


def normal_map(h, strength):
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * strength
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * strength
    nz = np.ones_like(h)
    ln = np.sqrt(gx * gx + gy * gy + nz * nz)
    # glTF/OpenGL tangent space: +Y points up, so the Y gradient is negated
    rgb = np.stack([-gx / ln, -gy / ln, nz / ln], -1)
    return ((rgb * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def blur(a, passes=3):
    k = np.array([1, 2, 1], dtype=np.float64); k /= k.sum()
    for _ in range(passes):
        for ax in (0, 1):
            a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), ax, a)
    return a


def srgb(lin):
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, 1) ** (1 / 2.4) - 0.055)



if WEAVE in ("plain", "twill", "satin"):
    h, ix, jy = interlaced_height(WEAVE if WEAVE != "satin" else "twill", THREADS)
    dye = rng.normal(1.0, 0.045, (THREADS, THREADS))    # per-yarn dye lot variance
    variance = dye[ix, jy]
elif WEAVE == "boucle":
    h = boucle_height()
    variance = 1.0 + blur(rng.normal(0, 0.09, (SIZE, SIZE)), 2)
else:
    h = velvet_height()
    variance = 1.0 + blur(rng.normal(0, 0.05, (SIZE, SIZE)), 1)

hn = (h - h.min()) / (np.ptp(h) + 1e-9)

Image.fromarray(normal_map(h, NSTR)).save(OUT / f"{SLUG}_normal.png", optimize=True)

# The base colour carries NO dye. The studio tints it, and a map with a colour already
# in it multiplies against that tint and turns every colourway to mud. Mean 0.85 rather
# than pure white leaves headroom for the yarn crowns instead of clipping them flat.
shade = 0.74 + 0.26 * hn                                # yarn crowns catch light
lin = np.clip(shade * variance, 0, 1)
lin *= 0.85 / max(float(lin.mean()), 1e-6)
grey = srgb(np.clip(lin, 0, 1))
Image.fromarray((np.stack([grey] * 3, -1) * 255).astype(np.uint8)).save(
    OUT / f"{SLUG}_weave.jpg", quality=92, subsampling=0)

ao = np.clip(0.55 + 0.45 * blur(hn, 2), 0, 1)           # valleys self-occlude
rg = np.clip(ROUGH - 0.10 * hn, 0, 1)                   # crowns slightly smoother
orm = np.stack([ao, rg, np.zeros_like(ao)], -1)         # B=metalness=0, fabric is dielectric
Image.fromarray((orm * 255).astype(np.uint8)).save(
    OUT / f"{SLUG}_orm.jpg", quality=92, subsampling=0)

print(f"{SLUG} {WEAVE} threads={THREADS} rough={ROUGH} normal={NSTR} -> {OUT}")
