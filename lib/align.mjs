/**
 * Which way is the reconstruction facing, and does it line up with the photograph?
 *
 * Hunyuan3D returns no camera. It reconstructs into a canonical frame that is roughly
 * aligned with the input view, but "roughly" is not something to build a texture bake
 * on faith, so this measures it instead of assuming it.
 *
 * The measurement is silhouette agreement. The matte already produced a subject mask
 * for the photograph, and a mesh rendered from a candidate camera produces another. If
 * the camera is right the two masks sit on top of each other; if it is wrong they do
 * not. Intersection over union turns that into one number per candidate, so picking a
 * camera becomes picking a maximum rather than eyeballing a render.
 *
 * Orthographic, not perspective. A product photograph is usually taken far enough back
 * and long enough in the lens that the parallax across the object is small, and an
 * orthographic fit has two fewer unknowns to go wrong.
 */

/**
 * Rasterise a mesh silhouette from a candidate view into a size x size bitmask.
 *
 * `fit` places the projected shape in the frame: scale is a multiple of the
 * bounding-box fit, dx/dy shift it in fractions of the frame. Bounding-box fitting
 * alone is not enough, because the photograph's mask carries the contact shadow and
 * the mesh does not, so the two boxes describe different objects.
 */
export function silhouette(positions, indices, view, size = 192, fit = { scale: 1, dx: 0, dy: 0 }) {
  const mask = new Uint8Array(size * size);
  const n = indices.length;

  // The projection depends on the view only; scale and offset are affine in pixel
  // space. The placement search evaluates hundreds of fits per view, so the
  // projection is computed once per view and kept on it.
  let proj = view._proj && view._proj.positions === positions ? view._proj : null;
  if (!proj) {
    const px = new Float32Array(positions.length / 3);
    const py = new Float32Array(positions.length / 3);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
      const [x, y, z] = [positions[i], positions[i + 1], positions[i + 2]];
      const u = view.right[0] * x + view.right[1] * y + view.right[2] * z;
      const v = view.up[0] * x + view.up[1] * y + view.up[2] * z;
      px[i / 3] = u; py[i / 3] = v;
      if (u < minX) minX = u; if (u > maxX) maxX = u;
      if (v < minY) minY = v; if (v > maxY) maxY = v;
    }
    proj = view._proj = { positions, px, py, minX, maxX, minY, maxY };
  }
  const { px, py, minX, maxX, minY, maxY } = proj;
  const span = (Math.max(maxX - minX, maxY - minY) || 1) / fit.scale;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const kx = (size - 1) / span, ox = (0.5 + fit.dx) * (size - 1) - cx * kx;
  const ky = -(size - 1) / span, oy = (0.5 + fit.dy) * (size - 1) - cy * ky;

  const a = [0, 0], b = [0, 0], c = [0, 0];
  for (let t = 0; t < n; t += 3) {
    const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
    const ax = px[i0] * kx + ox, ay = py[i0] * ky + oy;
    const bx = px[i1] * kx + ox, by = py[i1] * ky + oy;
    const cxp = px[i2] * kx + ox, cyp = py[i2] * ky + oy;
    a[0] = ax; a[1] = ay; b[0] = bx; b[1] = by; c[0] = cxp; c[1] = cyp;

    const lo = Math.max(0, Math.floor(Math.min(ax, bx, cxp)));
    const hi = Math.min(size - 1, Math.ceil(Math.max(ax, bx, cxp)));
    const vlo = Math.max(0, Math.floor(Math.min(ay, by, cyp)));
    const vhi = Math.min(size - 1, Math.ceil(Math.max(ay, by, cyp)));
    const area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    if (Math.abs(area) < 1e-9) continue;

    for (let y = vlo; y <= vhi; y++) {
      for (let x = lo; x <= hi; x++) {
        // barycentric sign test, no winding assumption: a silhouette wants both faces
        const w0 = ((b[0] - a[0]) * (y + 0.5 - a[1]) - (x + 0.5 - a[0]) * (b[1] - a[1])) / area;
        const w1 = ((c[0] - b[0]) * (y + 0.5 - b[1]) - (x + 0.5 - b[0]) * (c[1] - b[1])) / area;
        const w2 = ((a[0] - c[0]) * (y + 0.5 - c[1]) - (x + 0.5 - c[0]) * (a[1] - c[1])) / area;
        if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
          mask[y * size + x] = 1;
        }
      }
    }
  }
  return mask;
}

/**
 * Intersection over the smaller area. Tolerant of a part that is elsewhere in one of
 * the two shapes: a figure whose tail is raised in the model and extended in the
 * photograph scores about the body's share on this, and badly on IoU, which counts
 * both tails against it.
 */
export function overlap(a, b) {
  let inter = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { if (a[i]) na++; if (b[i]) nb++; if (a[i] && b[i]) inter++; }
  // the model may not shrink into the mask: its area must stay within reason of it,
  // or the placement search wins by hiding a small model inside a big mask
  if (!na || !nb || na < 0.6 * nb || na > 1.5 * nb) return 0;
  return inter / na;
}

export function iou(a, b) {
  let inter = 0, union = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] || b[i]) union++;
    if (a[i] && b[i]) inter++;
  }
  return union ? inter / union : 0;
}

/**
 * The candidate cameras.
 *
 * Six axis-aligned views plus a yaw sweep around the vertical, which is where the
 * uncertainty actually lives: the model reliably puts the object upright, and what is
 * not reliable is which way round it ended up.
 */
/** One candidate camera: yaw about the vertical, then tilted down by pitch. */
export function viewFor(deg, pitch) {
  const y = (deg * Math.PI) / 180;
  const p = (pitch * Math.PI) / 180;
  const forward = [Math.sin(y) * Math.cos(p), -Math.sin(p), Math.cos(y) * Math.cos(p)];
  const right = [Math.cos(y), 0, -Math.sin(y)];
  const up = [
    forward[1] * right[2] - forward[2] * right[1],
    forward[2] * right[0] - forward[0] * right[2],
    forward[0] * right[1] - forward[1] * right[0],
  ];
  return { name: `yaw ${deg} pitch ${pitch}`, yaw: deg, pitch, right, up, forward };
}

export function candidates(yawStep = 10, pitches = [0, 10, 20, 30, 40]) {
  const list = [];
  for (let deg = 0; deg < 360; deg += yawStep) {
    for (const pitch of pitches) {
      const y = (deg * Math.PI) / 180;
      const p = (pitch * Math.PI) / 180;
      // yaw about the vertical, then tilt down by pitch. Product photography almost
      // always looks slightly down at the object, and leaving pitch out of the search
      // was why nothing scored: the true camera was not in the candidate set at all.
      const forward = [Math.sin(y) * Math.cos(p), -Math.sin(p), Math.cos(y) * Math.cos(p)];
      const right = [Math.cos(y), 0, -Math.sin(y)];
      // up = forward x right, which points +y at pitch 0. It was right x forward for
      // a long time, which points DOWN, so every silhouette was rendered upside down
      // and scored against an upright photograph. A chair is close enough to
      // top-bottom symmetric that the scores looked plausible (0.73); a standing
      // figure was not: its own front scored 0.28 upright and 0.81 flipped.
      const up = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
      ];
      list.push({ name: `yaw ${deg} pitch ${pitch}`, yaw: deg, pitch, right, up, forward });
    }
  }
  return list;
}

/**
 * Find the view, and the placement within it, that best matches the photograph.
 *
 * Coarse over orientation, then a local hill climb over scale and offset for the few
 * best orientations. The placement search matters as much as the angle: the mask from
 * the photograph includes the contact shadow, so the two shapes never share a bounding
 * box and a pure orientation search tops out well short of a usable fit.
 */
export function findView(positions, indices, photoMask, size = 192, step = 10, metric = iou) {
  // Coarse to fine. The coarse sweep runs at half resolution, which is a quarter of
  // the raster work per silhouette, in 20-degree steps; the best four are then
  // searched at 5 degrees around themselves at full resolution. Measured on the
  // figure: the flat 10-degree sweep took 21 s for 0.866, this takes less for 0.873.
  // Alignment runs once per photograph, so it is most of the wall clock.
  const half = size >> 1;
  const halfMask = new Uint8Array(half * half);
  for (let y = 0; y < half; y++) for (let x = 0; x < half; x++) halfMask[y * half + x] = photoMask[(y * 2) * size + x * 2];
  const coarse = candidates(Math.max(step, 20)).map((view) => ({
    view,
    score: metric(silhouette(positions, indices, view, half), halfMask),
    fit: { scale: 1, dx: 0, dy: 0 },
  })).sort((a, b) => b.score - a.score);

  const seen = new Set();
  const fine = [];
  for (const c of coarse.slice(0, 4)) {
    for (const dy of [-10, -5, 0, 5, 10]) {
      for (const dp of [-5, 0, 5]) {
        const yaw = (c.view.yaw + dy + 360) % 360, pitch = c.view.pitch + dp;
        if (pitch < 0 || pitch > 50) continue;
        const key = `${yaw},${pitch}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const view = viewFor(yaw, pitch);
        fine.push({ view, score: metric(silhouette(positions, indices, view, size), photoMask) });
      }
    }
  }
  fine.sort((a, b) => b.score - a.score);

  const refined = fine.slice(0, 3).map(({ view }) => {
    let best = { scale: 1, dx: 0, dy: 0 };
    let bestScore = metric(silhouette(positions, indices, view, size, best), photoMask);
    for (let stepSize of [0.16, 0.08, 0.04, 0.02, 0.01]) {
      let improved = true;
      while (improved) {
        improved = false;
        for (const d of [
          { scale: stepSize }, { scale: -stepSize },
          { dx: stepSize }, { dx: -stepSize },
          { dy: stepSize }, { dy: -stepSize },
        ]) {
          const trial = {
            scale: best.scale + (d.scale || 0),
            dx: best.dx + (d.dx || 0),
            dy: best.dy + (d.dy || 0),
          };
          if (trial.scale < 0.3 || trial.scale > 3) continue;
          const sc = metric(silhouette(positions, indices, view, size, trial), photoMask);
          if (sc > bestScore + 1e-4) { bestScore = sc; best = trial; improved = true; }
        }
      }
    }
    return { view, fit: best, score: bestScore };
  }).sort((a, b) => b.score - a.score);

  return { best: refined[0], all: refined, coarse };
}
