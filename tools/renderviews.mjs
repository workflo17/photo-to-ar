// Software-render a coloured blank from the four canonical sides.
//
// Purpose: manufacture geometrically HONEST multi-view inputs. The earlier A/B used a
// mirrored front as the "right" view, which isolates the checkpoint effect but cannot
// show multi-view's real value, because a mirrored front is not what a right view
// looks like. Rendering an actual 3D mesh from four angles gives four views that are
// consistent by construction, so a reconstruction from them can be scored against the
// very mesh they came from.
//
// Plain z-buffered rasteriser with Lambert shading over vertex colours. No GL, no
// dependency: the same barycentric machinery as lib/atlas.mjs plus a depth test.
//
// Usage: node tools/renderviews.mjs <project-dir> <outdir> [size]
import { mkdir } from 'node:fs/promises';
import { NodeIO } from '@gltf-transform/core';
import sharp from 'sharp';

const [dir, outDir, sizeArg, flatArg] = process.argv.slice(2);
const SIZE = Number(sizeArg) || 1024;
// --flat renders in matte white, which shows surface quality the colour would hide
const FLAT = flatArg === '--flat';
await mkdir(outDir, { recursive: true });

const doc = await new NodeIO().read(`${dir}/blank.glb`);
const prim = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
const pos = prim.getAttribute('POSITION').getArray();
const idx = prim.getIndices().getArray();
let col = FLAT ? null : prim.getAttribute('COLOR_0')?.getArray();
if (!col) col = new Float32Array(pos.length).fill(0.78);   // matte white, linear

const VIEWS = {
  front: 0, right: 90, back: 180, left: 270,
};

const toSrgb = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);

for (const [name, deg] of Object.entries(VIEWS)) {
  const r = (deg * Math.PI) / 180;
  // orthographic basis looking at the object from `deg` around the vertical
  const right = [Math.cos(r), 0, -Math.sin(r)];
  const up = [0, 1, 0];
  const fwd = [Math.sin(r), 0, Math.cos(r)];   // towards the camera

  const n = pos.length / 3;
  const px = new Float32Array(n), py = new Float32Array(n), pz = new Float32Array(n);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const [x, y, z] = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    px[i] = right[0] * x + right[1] * y + right[2] * z;
    py[i] = up[0] * x + up[1] * y + up[2] * z;
    pz[i] = fwd[0] * x + fwd[1] * y + fwd[2] * z;
    if (px[i] < minX) minX = px[i]; if (px[i] > maxX) maxX = px[i];
    if (py[i] < minY) minY = py[i]; if (py[i] > maxY) maxY = py[i];
  }
  // fit with a margin, preserving aspect, like a product photograph
  const span = Math.max(maxX - minX, maxY - minY) * 1.15;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const X = (i) => ((px[i] - cx) / span + 0.5) * (SIZE - 1);
  const Y = (i) => (0.5 - (py[i] - cy) / span) * (SIZE - 1);

  const rgb = new Uint8ClampedArray(SIZE * SIZE * 3).fill(255);   // white backdrop
  const zbuf = new Float32Array(SIZE * SIZE).fill(-Infinity);

  for (let t = 0; t < idx.length; t += 3) {
    const i0 = idx[t], i1 = idx[t + 1], i2 = idx[t + 2];
    const ax = X(i0), ay = Y(i0), bx = X(i1), by = Y(i1), cxp = X(i2), cyp = Y(i2);
    const area = (bx - ax) * (cyp - ay) - (cxp - ax) * (by - ay);
    if (Math.abs(area) < 1e-9) continue;

    // face normal in view space, for Lambert; light sits at the camera
    const e1 = [px[i1] - px[i0], py[i1] - py[i0], pz[i1] - pz[i0]];
    const e2 = [px[i2] - px[i0], py[i2] - py[i0], pz[i2] - pz[i0]];
    const nz = e1[0] * e2[1] - e1[1] * e2[0];
    const nl = Math.hypot(
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      nz,
    ) || 1;
    // headlight shading, softened so unlit faces stay readable, as fill light would
    const lambert = 0.35 + 0.65 * Math.abs(nz / nl);

    const lo = Math.max(0, Math.floor(Math.min(ax, bx, cxp)));
    const hi = Math.min(SIZE - 1, Math.ceil(Math.max(ax, bx, cxp)));
    const vlo = Math.max(0, Math.floor(Math.min(ay, by, cyp)));
    const vhi = Math.min(SIZE - 1, Math.ceil(Math.max(ay, by, cyp)));

    for (let y = vlo; y <= vhi; y++) {
      for (let x = lo; x <= hi; x++) {
        const w0 = ((bx - ax) * (y + 0.5 - ay) - (x + 0.5 - ax) * (by - ay)) / area;
        const w1 = ((cxp - bx) * (y + 0.5 - by) - (x + 0.5 - bx) * (cyp - by)) / area;
        const w2 = 1 - w0 - w1;
        if ((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0)) continue;
        // weights are opposite-vertex: w1->i0, w2->i1, w0->i2
        const depth = w1 * pz[i0] + w2 * pz[i1] + w0 * pz[i2];
        const o = y * SIZE + x;
        if (depth <= zbuf[o]) continue;
        zbuf[o] = depth;
        for (let k = 0; k < 3; k++) {
          const lin = (w1 * col[i0 * 3 + k] + w2 * col[i1 * 3 + k] + w0 * col[i2 * 3 + k]) * lambert;
          rgb[o * 3 + k] = 255 * toSrgb(Math.min(1, Math.max(0, lin)));
        }
      }
    }
  }

  await sharp(Buffer.from(rgb.buffer), { raw: { width: SIZE, height: SIZE, channels: 3 } })
    .png().toFile(`${outDir}/${name}.png`);
  console.log(`${name}.png (yaw ${deg})`);
}
