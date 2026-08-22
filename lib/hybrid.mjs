/**
 * Shape from one engine, colour from the other.
 *
 * Hunyuan3D, fed every photograph, gets the shape right where a single photograph
 * cannot: the dorsal plates on a figure's back come out as jagged plates instead of
 * the generic row of knobs the single-view colour engine invents. But Hunyuan3D
 * returns no colour. TripoSplat returns the product's own colour and a worse shape.
 * So: take the shape from one and the colour from the other.
 *
 * The two meshes sit in different canonical frames (the same photograph came back
 * facing +x from one model and roughly +z from the other), so the splat has to be
 * registered onto the Hunyuan mesh first. Both are normalised to unit height about
 * their bounding-box centre, then a yaw is searched that makes their silhouettes
 * agree from four directions, then the splat's bounding box is matched per axis.
 * Colour transfers by nearest vertex in that frame.
 */
import { NodeIO } from '@gltf-transform/core';
import { silhouette, iou, viewFor } from './align.mjs';

function bounds(pos, idx) {
  const used = new Uint8Array(pos.length / 3);
  for (let i = 0; i < idx.length; i++) used[idx[i]] = 1;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length / 3; i++) {
    if (!used[i]) continue;
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], pos[i * 3 + k]); max[k] = Math.max(max[k], pos[i * 3 + k]); }
  }
  return { min, max, centre: min.map((m, k) => (m + max[k]) / 2), size: max.map((m, k) => m - min[k]) };
}

/** Normalise to unit height about the bbox centre, optionally yawed about y. */
function normalised(pos, idx, yawDeg = 0) {
  const b = bounds(pos, idx);
  const s = 1 / (b.size[1] || 1);
  const c = Math.cos(yawDeg * Math.PI / 180), sn = Math.sin(yawDeg * Math.PI / 180);
  const out = new Float32Array(pos.length);
  for (let i = 0; i < pos.length; i += 3) {
    const x = (pos[i] - b.centre[0]) * s, y = (pos[i + 1] - b.centre[1]) * s, z = (pos[i + 2] - b.centre[2]) * s;
    out[i] = c * x + sn * z; out[i + 1] = y; out[i + 2] = -sn * x + c * z;
  }
  return out;
}

/**
 * Find the yaw that turns the splat into the Hunyuan frame.
 * Scored by silhouette agreement from four directions plus one from above.
 */
export function registerYaw(hunyuan, splat, log = () => {}) {
  const views = [viewFor(0, 10), viewFor(90, 10), viewFor(180, 10), viewFor(270, 10), viewFor(45, 45)];
  const hn = normalised(hunyuan.positions, hunyuan.indices);
  const targets = views.map((v) => silhouette(hn, hunyuan.indices, v, 160));
  let best = { yaw: 0, score: -1 };
  const scoreOf = (yaw) => {
    const sp = normalised(splat.positions, splat.indices, yaw);
    let s = 0;
    for (let i = 0; i < views.length; i++) {
      const v = { ...views[i] }; delete v._proj;
      s += iou(silhouette(sp, splat.indices, v, 160), targets[i]);
    }
    return s / views.length;
  };
  for (let yaw = 0; yaw < 360; yaw += 10) { const s = scoreOf(yaw); if (s > best.score) best = { yaw, score: s }; }
  for (const d of [-6, -4, -2, 2, 4, 6]) { const yaw = (best.yaw + d + 360) % 360; const s = scoreOf(yaw); if (s > best.score) best = { yaw, score: s }; }
  log(`registered the colour mesh onto the shape: yaw ${best.yaw} degrees, silhouettes agree ${(best.score * 100).toFixed(0)}%`);
  return best;
}

/**
 * Paint the Hunyuan mesh with the splat's colour. Returns a COLOR_0 array for the
 * Hunyuan vertices, plus how far the nearest colour was on average.
 */
export function transferColour(hunyuan, splat, yaw) {
  const hn = normalised(hunyuan.positions, hunyuan.indices);
  const sp = normalised(splat.positions, splat.indices, yaw);
  // match the splat's box to the Hunyuan box per axis, after the yaw
  const hb = bounds(hn, hunyuan.indices), sb = bounds(sp, splat.indices);
  const scale = hb.size.map((s, k) => s / (sb.size[k] || 1));
  for (let i = 0; i < sp.length; i += 3) for (let k = 0; k < 3; k++) sp[i + k] = (sp[i + k] - sb.centre[k]) * scale[k] + hb.centre[k];

  const cell = 0.02;   // of unit height
  const grid = new Map();
  const keyOf = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  const usedS = new Uint8Array(sp.length / 3);
  for (let i = 0; i < splat.indices.length; i++) usedS[splat.indices[i]] = 1;
  for (let i = 0; i < sp.length / 3; i++) {
    if (!usedS[i]) continue;
    const k = keyOf(sp[i * 3], sp[i * 3 + 1], sp[i * 3 + 2]);
    let l = grid.get(k); if (!l) grid.set(k, l = []); l.push(i);
  }
  const n = hn.length / 3;
  const out = new Float32Array(n * 3);
  let distSum = 0, found = 0;
  let mr = 0, mg = 0, mb = 0, mc = 0;
  for (let i = 0; i < sp.length / 3; i++) if (usedS[i]) { mr += splat.colours[i * 3]; mg += splat.colours[i * 3 + 1]; mb += splat.colours[i * 3 + 2]; mc++; }
  const fallback = [mr / mc, mg / mc, mb / mc];
  for (let v = 0; v < n; v++) {
    const x = hn[v * 3], y = hn[v * 3 + 1], z = hn[v * 3 + 2];
    const gx = Math.floor(x / cell), gy = Math.floor(y / cell), gz = Math.floor(z / cell);
    let best = -1, bestD = Infinity;
    for (let r = 1; r <= 3 && best < 0; r++) {
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        const l = grid.get(`${gx + dx},${gy + dy},${gz + dz}`); if (!l) continue;
        for (const j of l) {
          const d = (sp[j * 3] - x) ** 2 + (sp[j * 3 + 1] - y) ** 2 + (sp[j * 3 + 2] - z) ** 2;
          if (d < bestD) { bestD = d; best = j; }
        }
      }
    }
    if (best >= 0) {
      for (let k = 0; k < 3; k++) out[v * 3 + k] = splat.colours[best * 3 + k];
      distSum += Math.sqrt(bestD); found++;
    } else {
      for (let k = 0; k < 3; k++) out[v * 3 + k] = fallback[k];
    }
  }
  return { colours: out, meanDistance: found ? distSum / found : Infinity, coverage: found / n };
}

/**
 * Read both raw meshes, register, transfer, write the hybrid raw mesh (Hunyuan
 * shape, splat colour) to outPath. Returns the registration facts.
 */
export async function hybridRaw(shapePath, colourPath, outPath, log = () => {}) {
  const io = new NodeIO();
  const shapeDoc = await io.read(shapePath);
  const colourDoc = await io.read(colourPath);
  const sp = shapeDoc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
  const cp = colourDoc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
  const hunyuan = { positions: sp.getAttribute('POSITION').getArray(), indices: sp.getIndices().getArray() };
  const splat = {
    positions: cp.getAttribute('POSITION').getArray(), indices: cp.getIndices().getArray(),
    colours: cp.getAttribute('COLOR_0')?.getArray(),
  };
  if (!splat.colours) throw new Error('the colour mesh has no vertex colour');
  const reg = registerYaw(hunyuan, splat, log);
  const t = transferColour(hunyuan, splat, reg.yaw);
  log(`colour transferred to ${(hunyuan.positions.length / 3).toLocaleString()} vertices, nearest colour ${(t.meanDistance * 100).toFixed(2)}% of the height away on average`);
  sp.setAttribute('COLOR_0', shapeDoc.createAccessor().setType('VEC3').setArray(t.colours).setBuffer(shapeDoc.getRoot().listBuffers()[0]));
  await io.write(outPath, shapeDoc);
  return { yaw: reg.yaw, agreement: reg.score, meanDistance: t.meanDistance };
}
