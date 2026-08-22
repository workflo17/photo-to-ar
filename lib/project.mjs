/**
 * Put the photographs' own pixels on the model.
 *
 * The colour engine recovers colour from one photograph, and everything that
 * photograph did not see is invented: the flanks and back of a figure come out as
 * blotches of plausible grey. When more photographs exist, the honest colour is in
 * them. This module aligns each photograph to the model by silhouette (lib/align.mjs),
 * projects it onto the colour atlas where the surface faces that camera and is not
 * hidden, and blends it with what is already there.
 *
 * Orthographic cameras, like the alignment, and the alignment is only as good as
 * the silhouette match: a pose that differs between shots (a tail that moved) lands
 * colour a little off. Three things limit the damage. The mask is eroded so edge
 * pixels, the ones most likely to be backdrop, are never sampled. Each photograph's
 * exposure is matched to the existing colour over the texels both cover, so a darker
 * shot does not paint a darker stripe. And the existing colour keeps its weight
 * where the surface faced the reconstruction's own camera, where it is best.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';
import { NodeIO } from '@gltf-transform/core';
import { findView, silhouette, iou, overlap } from './align.mjs';
import { cubeAtlas } from './atlas.mjs';

const SIDES = ['front', 'left', 'back', 'right'];

/** Decode a mask PNG to one byte per pixel, whatever channel count sharp returns. */
export async function decodeMask(png) {
  const { data, info } = await sharp(png).toColourspace('b-w').raw().toBuffer({ resolveWithObject: true });
  const out = new Uint8Array(info.width * info.height);
  for (let i = 0; i < out.length; i++) out[i] = data[i * info.channels] > 127 ? 1 : 0;
  return { mask: out, width: info.width, height: info.height };
}

/** Letterbox the subject's bounding box into a 192 square, as lib/photomask.py does. */
export async function mask192(m) {
  let top = 1e9, bot = -1, left = 1e9, right = -1;
  for (let y = 0; y < m.height; y++) for (let x = 0; x < m.width; x++) {
    if (!m.mask[y * m.width + x]) continue;
    if (y < top) top = y; if (y > bot) bot = y; if (x < left) left = x; if (x > right) right = x;
  }
  if (bot < 0) return null;
  const bw = right - left + 1, bh = bot - top + 1, span = Math.max(bw, bh);
  const sq = Buffer.alloc(span * span);
  const ox = (span - bw) >> 1, oy = (span - bh) >> 1;
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) sq[(y + oy) * span + x + ox] = m.mask[(top + y) * m.width + left + x] ? 255 : 0;
  const r = await sharp(sq, { raw: { width: span, height: span, channels: 1 } }).resize(192, 192).raw().toBuffer({ resolveWithObject: true });
  const small = new Uint8Array(192 * 192);
  for (let i = 0; i < small.length; i++) small[i] = r.data[i * r.info.channels] > 127 ? 1 : 0;
  return { small, box: { left, top, bw, bh, span, ox, oy } };
}

/** Erode a mask by r pixels (square), so the sampled region stays clear of its edge. */
function erode(m, r) {
  const out = new Uint8Array(m.mask.length);
  const { width: w, height: h, mask } = m;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue;
    let ok = 1;
    for (let dy = -r; dy <= r && ok; dy++) for (let dx = -r; dx <= r; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w || !mask[yy * w + xx]) { ok = 0; break; }
    }
    out[y * w + x] = ok;
  }
  return { ...m, mask: out };
}

/**
 * @param dir       project directory (blank.glb, atlas.bin, photo.jpg, source-<side>.png)
 * @param maskFor   async (photoBuffer) => mask PNG buffer
 * @param log       progress callback
 * @returns {Array} one record per photograph used: side, fit score, texels painted
 */
export async function projectPhotos(dir, { maskFor, log = () => {}, minFit = 0.6, known = {} } = {}) {
  const doc = await new NodeIO().read(join(dir, 'blank.glb'));
  const prim = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
  const positions = prim.getAttribute('POSITION').getArray();
  const normals = prim.getAttribute('NORMAL').getArray();
  const indices = prim.getIndices().getArray();
  // the parts, for pose-tolerant alignment: a part that moved between shots (a
  // tail) can be left out of the silhouette and out of the painting
  let perTriangle = null, regionShare = [];
  try {
    const rj = JSON.parse(await readFile(join(dir, 'regions.json'), 'utf8'));
    perTriangle = new Uint16Array(Buffer.from(rj.perTriangle, 'base64').buffer);
    regionShare = rj.regions.map((r) => r.share || 0);
  } catch { /* no parts: whole-silhouette alignment only */ }
  const atlasRaw = await readFile(join(dir, 'atlas.bin'));
  const uvs = new Float32Array(atlasRaw.buffer, atlasRaw.byteOffset, atlasRaw.byteLength / 4);
  // unwelded per-corner positions and normals, in the atlas's own order
  const at = cubeAtlas(positions, normals, null, indices);

  // the existing texture, as linear floats, plus a weight per texel
  const tex = await sharp(join(dir, 'photo.jpg')).raw().toBuffer({ resolveWithObject: true });
  const size = tex.info.width;   // the atlas decides the resolution, whatever it is
  const toLin = new Float32Array(256);
  for (let i = 0; i < 256; i++) { const c = i / 255; toLin[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }
  const ch = tex.info.channels;
  const acc = new Float32Array(size * size * 3);
  const wsum = new Float32Array(size * size);
  const seenBest = new Float32Array(size * size);   // best facing any accepted photo had
  // per-texel luminance range across photographs that saw it well: gloss shows as
  // the same point reading brighter from one angle than another
  const lumMin = new Float32Array(size * size).fill(Infinity);
  const lumMax = new Float32Array(size * size);
  const lumN = new Uint8Array(size * size);
  const base = new Float32Array(size * size * 3);
  for (let i = 0; i < size * size; i++) for (let k = 0; k < 3; k++) base[i * 3 + k] = toLin[tex.data[i * ch + k]];

  // Per texel: world position and normal, from the atlas triangles. Rasterised once.
  const texPos = new Float32Array(size * size * 3);
  const texNrm = new Float32Array(size * size * 3);
  const covered = new Uint8Array(size * size);
  const texRegion = new Uint16Array(size * size);
  {
    const px = new Float32Array(uvs.length / 2), py = new Float32Array(uvs.length / 2);
    for (let i = 0; i < uvs.length / 2; i++) { px[i] = uvs[i * 2] * (size - 1); py[i] = uvs[i * 2 + 1] * (size - 1); }   // glTF convention, see atlas.mjs
    for (let t = 0; t < at.indices.length; t += 3) {
      const i0 = at.indices[t], i1 = at.indices[t + 1], i2 = at.indices[t + 2];
      const ax = px[i0], ay = py[i0], bx = px[i1], by = py[i1], cx = px[i2], cy = py[i2];
      const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
      if (Math.abs(area) < 1e-9) continue;
      const lo = Math.max(0, Math.floor(Math.min(ax, bx, cx))), hi = Math.min(size - 1, Math.ceil(Math.max(ax, bx, cx)));
      const vlo = Math.max(0, Math.floor(Math.min(ay, by, cy))), vhi = Math.min(size - 1, Math.ceil(Math.max(ay, by, cy)));
      for (let y = vlo; y <= vhi; y++) for (let x = lo; x <= hi; x++) {
        const w0 = ((bx - ax) * (y + 0.5 - ay) - (x + 0.5 - ax) * (by - ay)) / area;
        const w1 = ((cx - bx) * (y + 0.5 - by) - (x + 0.5 - bx) * (cy - by)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < -0.002 || w1 < -0.002 || w2 < -0.002) continue;
        const o = y * size + x;
        for (let k = 0; k < 3; k++) {
          texPos[o * 3 + k] = w1 * at.positions[i0 * 3 + k] + w2 * at.positions[i1 * 3 + k] + w0 * at.positions[i2 * 3 + k];
          texNrm[o * 3 + k] = w1 * at.normals[i0 * 3 + k] + w2 * at.normals[i1 * 3 + k] + w0 * at.normals[i2 * 3 + k];
        }
        const l = Math.hypot(texNrm[o * 3], texNrm[o * 3 + 1], texNrm[o * 3 + 2]) || 1;
        texNrm[o * 3] /= l; texNrm[o * 3 + 1] /= l; texNrm[o * 3 + 2] /= l;
        covered[o] = 1;
        if (perTriangle) texRegion[o] = perTriangle[t / 3];
      }
    }
  }

  const used = [];
  const refused = [];
  const silhouettesOut = [];
  let frontView = null;
  // which views may paint: views.json when the build wrote one, the cardinal four
  // otherwise; a shape-only view (another colourway of the same mould) stays out
  let sides = SIDES;
  try {
    const roles = JSON.parse(await readFile(join(dir, 'views.json'), 'utf8')).sides || {};
    const named = Object.keys(roles).filter((k) => roles[k].colour !== false);
    if (named.length) sides = ['front', ...named.filter((k) => k !== 'front')];
  } catch { /* no roles recorded: the cardinal four */ }
  for (const side of sides) {
    let photo;
    try { photo = await readFile(join(dir, `source-${side}.png`)); } catch { continue; }
    const m = await decodeMask(await maskFor(photo));
    const m192 = await mask192(m);
    if (!m192) { log(`${side}: no subject found`); refused.push({ side, reason: 'no subject found' }); continue; }
    // the fit stage already aligned the front; do not search it twice
    let best = known[side] || findView(positions, indices, m192.small, 192, 10).best;
    if (side === 'front') frontView = best.view;
    // Pose tolerance, by part. When the whole silhouette fails, ask which parts of the
    // model the photograph's mask covers at that best view. A part the mask barely
    // covers is a part posed differently in this shot (the tail, usually). Leave
    // those parts out, align the rest again, and if the rest fits, paint the rest:
    // the body of a figure from a photograph whose tail moved is still the body.
    let excluded = new Set();
    let tolerant = false;
    if (best.score < minFit) {
      // second chance: align on what the two shapes share rather than on everything
      // either has. Accepted when they share most of the smaller shape and the plain
      // fit at that view is at least a plausible one; the exposure test and the
      // per-part check below still stand between it and the texture.
      const alt = findView(positions, indices, m192.small, 192, 10, overlap).best;
      const plain = iou(silhouette(positions, indices, alt.view, 192, alt.fit), m192.small);
      log(`${side}: shared-shape search: ${(alt.score * 100).toFixed(0)}% of the smaller shape at ${alt.view.name}, plain fit there ${(plain * 100).toFixed(0)}%`);
      if (alt.score >= 0.8 && plain >= 0.35) {
        log(`${side}: ${(best.score * 100).toFixed(0)}% as a whole; the shapes share ${(alt.score * 100).toFixed(0)}% of the smaller one at ${alt.view.name} (plain fit ${(plain * 100).toFixed(0)}%), aligning on that`);
        best = { ...alt, score: plain };
        tolerant = true;
      }
    }
    if ((best.score < minFit || tolerant) && perTriangle && regionShare.length > 1) {
      const mask = m192.small;
      const byRegion = new Map();
      for (let t = 0; t < indices.length / 3; t++) {
        const r = perTriangle[t];
        let l = byRegion.get(r); if (!l) byRegion.set(r, l = []);
        l.push(indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
      }
      const coverage = [];
      for (const [r, idx] of byRegion) {
        const sil = silhouette(positions, Uint32Array.from(idx), best.view, 192, best.fit);
        let inter = 0, area = 0;
        for (let i = 0; i < sil.length; i++) { if (sil[i]) { area++; if (mask[i]) inter++; } }
        coverage.push({ r, cov: area ? inter / area : 0, share: regionShare[r] || 0 });
      }
      coverage.sort((a, b) => a.cov - b.cov);
      let dropped = 0;
      for (const c of coverage) {
        if (c.cov >= 0.5 || dropped + c.share > 0.4) continue;
        excluded.add(c.r); dropped += c.share;
      }
      if (excluded.size) {
        const keep = [];
        for (let t = 0; t < indices.length / 3; t++) if (!excluded.has(perTriangle[t])) keep.push(indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
        const again = findView(positions, Uint32Array.from(keep), mask, 192, 10).best;
        if (again.score >= minFit) {
          log(`${side}: ${(again.score * 100).toFixed(0)}% without ${excluded.size} part${excluded.size === 1 ? '' : 's'} that moved (${(dropped * 100).toFixed(0)}% of the surface); painting the rest`);
          best = again;
        } else if (tolerant) {
          log(`${side}: leaving out ${excluded.size} part${excluded.size === 1 ? '' : 's'} the photograph does not cover (${(dropped * 100).toFixed(0)}% of the surface)`);
        } else {
          excluded = new Set();
        }
      }
    }
    if (best.score < minFit && !tolerant) {
      log(`${side}: silhouette match ${(best.score * 100).toFixed(0)}%, too far off to paint from`);
      refused.push({ side, fit: best.score, reason: 'silhouette does not match: the pose differs, or the model is wrong from this side' });
      continue;
    }
    const { view, fit } = best;

    // the frame the silhouette was fitted in, so a world point maps to a photo pixel
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
      const u = view.right[0] * positions[i] + view.right[1] * positions[i + 1] + view.right[2] * positions[i + 2];
      const v = view.up[0] * positions[i] + view.up[1] * positions[i + 1] + view.up[2] * positions[i + 2];
      if (u < minU) minU = u; if (u > maxU) maxU = u; if (v < minV) minV = v; if (v > maxV) maxV = v;
    }
    const span = (Math.max(maxU - minU, maxV - minV) || 1) / fit.scale;
    const cu = (minU + maxU) / 2, cv = (minV + maxV) / 2;
    const { box } = m192;
    const toPhoto = (p) => {
      const u = view.right[0] * p[0] + view.right[1] * p[1] + view.right[2] * p[2];
      const v = view.up[0] * p[0] + view.up[1] * p[1] + view.up[2] * p[2];
      const fx = (u - cu) / span + 0.5 + fit.dx;
      const fy = 0.5 - (v - cv) / span + fit.dy;
      return [box.left - box.ox + fx * box.span, box.top - box.oy + fy * box.span];
    };
    // depth along the view, nearer is smaller; which sign is "towards the camera" is
    // settled by the silhouette fit being orthographic: either sign gives the same
    // silhouette, so the convention is picked by the normals below (a surface facing
    // the camera has a normal against the forward vector)
    const fwd = view.forward;
    const depthOf = (p) => fwd[0] * p[0] + fwd[1] * p[1] + fwd[2] * p[2];

    // visibility: a depth buffer over the photo frame at the atlas's scale
    const Z = 1024;
    const zbuf = new Float32Array(Z * Z).fill(Infinity);
    const toZ = (p) => { const [x, y] = toPhoto(p); return [(x - box.left + box.ox) / box.span * Z, (y - box.top + box.oy) / box.span * Z]; };
    for (let t = 0; t < indices.length; t += 3) {
      const P = [0, 1, 2].map((k) => { const v = indices[t + k]; return [positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]]; });
      const S = P.map(toZ), D = P.map(depthOf);
      const [ax, ay] = S[0], [bx, by] = S[1], [cx, cy] = S[2];
      const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
      if (Math.abs(area) < 1e-9) continue;
      const lo = Math.max(0, Math.floor(Math.min(ax, bx, cx))), hi = Math.min(Z - 1, Math.ceil(Math.max(ax, bx, cx)));
      const vlo = Math.max(0, Math.floor(Math.min(ay, by, cy))), vhi = Math.min(Z - 1, Math.ceil(Math.max(ay, by, cy)));
      for (let y = vlo; y <= vhi; y++) for (let x = lo; x <= hi; x++) {
        const w0 = ((bx - ax) * (y + 0.5 - ay) - (x + 0.5 - ax) * (by - ay)) / area;
        const w1 = ((cx - bx) * (y + 0.5 - by) - (x + 0.5 - bx) * (cy - by)) / area;
        const w2 = 1 - w0 - w1;
        if ((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0)) continue;
        const d = w1 * D[0] + w2 * D[1] + w0 * D[2];
        const o = y * Z + x;
        if (d < zbuf[o]) zbuf[o] = d;
      }
    }

    // the photograph itself, linear, and its eroded mask
    const src = await sharp(photo).raw().toBuffer({ resolveWithObject: true });
    const sc = src.info.channels, sw = src.info.width, sh = src.info.height;
    const inner = erode(m, Math.max(2, Math.round(Math.min(sw, sh) / 120)));
    const tol = span / Z * 2.5;   // depth tolerance: a couple of depth-buffer texels

    // first pass: gather candidate colours and measure the exposure against the base
    const cand = new Float32Array(size * size * 3);
    const cw = new Float32Array(size * size);
    let gr = 0, gg = 0, gb = 0, br = 0, bg = 0, bb = 0, gn = 0;
    for (let o = 0; o < size * size; o++) {
      if (!covered[o]) continue;
      const p = [texPos[o * 3], texPos[o * 3 + 1], texPos[o * 3 + 2]];
      const n = [texNrm[o * 3], texNrm[o * 3 + 1], texNrm[o * 3 + 2]];
      const facing = -(n[0] * fwd[0] + n[1] * fwd[1] + n[2] * fwd[2]);
      if (facing < 0.3) continue;
      if (excluded.size && excluded.has(texRegion[o])) continue;
      const [x, y] = toPhoto(p);
      const xi = Math.round(x), yi = Math.round(y);
      if (xi < 0 || yi < 0 || xi >= sw || yi >= sh || !inner.mask[yi * sw + xi]) continue;
      const [zx, zy] = toZ(p);
      const zo = (zy | 0) * Z + (zx | 0);
      if (zx < 0 || zy < 0 || zx >= Z || zy >= Z || depthOf(p) > zbuf[zo] + tol) continue;
      // Sharp weights: facing^8, so where two photographs both see a texel the one
      // that faces it best wins almost outright. Averaging them at facing^2 turned
      // a shoe's Swoosh into a blur, because three photographs registered within a
      // few millimetres of each other are three slightly different Swooshes.
      const w = facing ** 8;
      for (let k = 0; k < 3; k++) cand[o * 3 + k] = toLin[src.data[(yi * sw + xi) * sc + k]];
      cw[o] = w;
      if (facing > 0.7) {
        gr += cand[o * 3]; gg += cand[o * 3 + 1]; gb += cand[o * 3 + 2];
        br += base[o * 3]; bg += base[o * 3 + 1]; bb += base[o * 3 + 2]; gn++;
      }
    }
    // De-light this photograph on its own terms. Its shading is a function of the
    // model's normals in this camera's frame: fit log-luminance against the normal
    // (an order-one spherical harmonic) over the texels that face the camera well,
    // and divide the fitted shading out when it explains enough of the variation.
    // A per-photo fit beats the one global fit on the vertex colours because each
    // photograph was lit by its own setup, and this is what keeps a surface painted
    // from two photographs from carrying two different key lights into AR.
    {
      const A = new Float64Array(16), B = new Float64Array(4);
      let n = 0, meanY = 0;
      const ys = [], xs = [];
      for (let o = 0; o < size * size; o++) {
        if (cw[o] < 0.5 ** 8) continue;   // facing > 0.5
        const L = 0.2126 * cand[o * 3] + 0.7152 * cand[o * 3 + 1] + 0.0722 * cand[o * 3 + 2];
        if (L < 0.004) continue;
        const y = Math.log(L);
        ys.push(y); xs.push([1, texNrm[o * 3], texNrm[o * 3 + 1], texNrm[o * 3 + 2]]);
        meanY += y; n++;
      }
      if (n > 500) {
        meanY /= n;
        let ssTot = 0;
        for (let i = 0; i < n; i++) {
          const x = xs[i];
          for (let r = 0; r < 4; r++) { B[r] += x[r] * ys[i]; for (let c = 0; c < 4; c++) A[r * 4 + c] += x[r] * x[c]; }
          ssTot += (ys[i] - meanY) ** 2;
        }
        // gaussian elimination, 4x4
        const M = Array.from({ length: 4 }, (_, r) => [...A.slice(r * 4, r * 4 + 4), B[r]]);
        let ok = true;
        for (let c = 0; c < 4 && ok; c++) {
          let piv = c; for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
          [M[c], M[piv]] = [M[piv], M[c]];
          if (Math.abs(M[c][c]) < 1e-12) { ok = false; break; }
          for (let r = 0; r < 4; r++) { if (r === c) continue; const f = M[r][c] / M[c][c]; for (let k = c; k < 5; k++) M[r][k] -= f * M[c][k]; }
        }
        if (ok) {
          const coef = M.map((row, r) => row[4] / row[r]);
          let ssRes = 0;
          for (let i = 0; i < n; i++) { const x = xs[i]; const pred = coef[0] + coef[1] * x[1] + coef[2] * x[2] + coef[3] * x[3]; ssRes += (ys[i] - pred) ** 2; }
          const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
          if (r2 >= 0.15) {
            let meanS = 0, cnt = 0;
            for (let o = 0; o < size * size; o++) { if (!cw[o]) continue; meanS += Math.exp(coef[1] * texNrm[o * 3] + coef[2] * texNrm[o * 3 + 1] + coef[3] * texNrm[o * 3 + 2]); cnt++; }
            meanS /= cnt || 1;
            const strength = 0.85;
            for (let o = 0; o < size * size; o++) {
              if (!cw[o]) continue;
              const sh = Math.exp(coef[1] * texNrm[o * 3] + coef[2] * texNrm[o * 3 + 1] + coef[3] * texNrm[o * 3 + 2]) / meanS;
              const f = 1 / (sh ** strength);
              for (let k = 0; k < 3; k++) cand[o * 3 + k] = Math.min(1, cand[o * 3 + k] * f);
            }
            const dl = Math.hypot(coef[1], coef[2], coef[3]) || 1;
            log(`${side}: de-lit, orientation explained ${(r2 * 100).toFixed(0)}% of its shading, light from [${[coef[1] / dl, coef[2] / dl, coef[3] / dl].map((v) => v.toFixed(2)).join(', ')}]`);
          } else {
            log(`${side}: shading not explained by orientation (${(r2 * 100).toFixed(0)}%), left as shot`);
          }
        }
      }
    }

    let painted = 0;
    for (let o = 0; o < size * size; o++) if (cw[o] > 0) painted++;
    if (process.env.PROJECT_DEBUG) {
      // the candidate colours as an image, and the photo with the sampled pixels marked
      const dbg = Buffer.alloc(size * size * 3);
      for (let o = 0; o < size * size; o++) for (let k = 0; k < 3; k++) dbg[o * 3 + k] = cw[o] ? Math.round(255 * Math.sqrt(cand[o * 3 + k])) : 0;
      await sharp(dbg, { raw: { width: size, height: size, channels: 3 } }).png().toFile(join(dir, `_debug-cand-${side}.png`));
      const marked = Buffer.from(src.data);
      for (let o = 0; o < size * size; o++) {
        if (!cw[o]) continue;
        const [x, y] = toPhoto([texPos[o * 3], texPos[o * 3 + 1], texPos[o * 3 + 2]]);
        const xi = Math.round(x), yi = Math.round(y);
        if (xi < 0 || yi < 0 || xi >= sw || yi >= sh) continue;
        marked[(yi * sw + xi) * sc] = 255; marked[(yi * sw + xi) * sc + 1] = 0;
      }
      await sharp(marked, { raw: { width: sw, height: sh, channels: sc } }).png().toFile(join(dir, `_debug-photo-${side}.png`));
    }
    if (!painted) { log(`${side}: aligned at ${(best.score * 100).toFixed(0)}% but nothing faces that camera`); refused.push({ side, fit: best.score, reason: 'nothing faces that camera' }); continue; }
    // exposure match, per channel, clamped: a photograph shot darker must not paint a
    // darker stripe, but a real colour difference must not be normalised away either
    let rawGain = gn ? [br / (gr || 1e-6), bg / (gg || 1e-6), bb / (gb || 1e-6)] : [1, 1, 1];
    if (best.score >= 0.85) {
      // A photograph placed this well is trusted over the base: the base under it
      // may be the reconstruction's guess (the splat paints an unseen back dark
      // grey; the real back is lit red). Match luminance only, gently, so hue and
      // saturation are the photograph's own.
      const lb = 0.2126 * br + 0.7152 * bg + 0.0722 * bb, lc = 0.2126 * gr + 0.7152 * gg + 0.0722 * gb;
      const g = gn ? Math.min(1.4, Math.max(0.7, lb / (lc || 1e-6))) : 1;
      rawGain = [g, g, g];
    } else if (rawGain.some((g) => g > 1.9 || g < 0.5)) {
      // At a looser fit a gain far from one means the texels that supposedly
      // correspond do not: a close-up letterboxed into a square can match a
      // silhouette it does not belong to, and the only sign is that the colours
      // need multiplying by 2.5 to agree.
      log(`${side}: aligned at ${(best.score * 100).toFixed(0)}% but its colours do not correspond (exposure x${rawGain.map((g) => g.toFixed(2)).join('/')}), skipped`);
      refused.push({ side, fit: best.score, reason: 'colours do not correspond: probably a different part of the product' });
      continue;
    }
    const gain = rawGain;
    for (let o = 0; o < size * size; o++) {
      if (!cw[o]) continue;
      for (let k = 0; k < 3; k++) acc[o * 3 + k] += cand[o * 3 + k] * gain[k] * cw[o];
      wsum[o] += cw[o];
      if (cw[o] > seenBest[o]) seenBest[o] = cw[o];
      if (cw[o] >= 0.5 ** 8) {   // facing > 0.5
        const L = 0.2126 * cand[o * 3] * gain[0] + 0.7152 * cand[o * 3 + 1] * gain[1] + 0.0722 * cand[o * 3 + 2] * gain[2];
        if (L < lumMin[o]) lumMin[o] = L;
        if (L > lumMax[o]) lumMax[o] = L;
        lumN[o]++;
      }
    }
    used.push({ side, fit: best.score, view: view.name, painted, gain: gain.map((g) => +g.toFixed(2)), excludedParts: [...excluded] });
    silhouettesOut.push({ side, fit, view, score: best.score, box, mask: m.mask, width: m.width, height: m.height,
      bounds: { minU, maxU, minV, maxV } });
    log(`${side}: silhouette match ${(best.score * 100).toFixed(0)}% at ${view.name}, painted ${painted.toLocaleString()} texels, exposure x${gain.map((g) => g.toFixed(2)).join('/')}`);
  }
  projectPhotos.lastRefused = refused;
  projectPhotos.lastSilhouettes = silhouettesOut;
  if (!used.length) return used;

  // Gloss from disagreement, measured before anything is shipped. A matte surface
  // reads the same brightness from every angle once exposure is matched; a glossy
  // one does not. The ratio max/min over the texels two or more photographs saw well
  // is the signal; its spread across the surface says whether it is a map or noise.
  let glossStats = null;
  {
    const ratios = [];
    for (let o = 0; o < size * size; o++) if (lumN[o] >= 2 && lumMin[o] > 0.01) ratios.push(lumMax[o] / lumMin[o]);
    if (ratios.length > 1000) {
      const sorted = Float32Array.from(ratios).sort();
      const q = (f) => sorted[Math.floor(sorted.length * f)];
      glossStats = { texels: ratios.length, p10: q(0.1), p50: q(0.5), p90: q(0.9) };
      log(`gloss: ${ratios.length.toLocaleString()} texels seen by two or more photographs, brightness ratio p10 ${q(0.1).toFixed(2)}, median ${q(0.5).toFixed(2)}, p90 ${q(0.9).toFixed(2)}`);
    }
  }
  projectPhotos.lastGloss = glossStats;

  // blend with the existing colour, which keeps its weight where it faced the
  // reconstruction's camera (the front photograph's view, when known)
  const blended = new Float32Array(size * size * 3);
  for (let o = 0; o < size * size; o++) {
    // the reconstruction's own colour keeps a say where it faced its camera, and
    // steps back wherever a photograph saw the texel well
    let wb = 0.35;
    if (frontView) {
      const f = -(texNrm[o * 3] * frontView.forward[0] + texNrm[o * 3 + 1] * frontView.forward[1] + texNrm[o * 3 + 2] * frontView.forward[2]);
      wb = Math.max(0.15, f * f);
    }
    // a photograph that faced this texel better than 0.5 is the authority on it;
    // the guess only survives where no photograph saw it that well
    const seenWell = Math.min(1, seenBest[o] / (0.5 ** 8));
    wb *= (1 - seenWell) + 0.01 * seenWell;
    const total = wb + wsum[o];
    for (let k = 0; k < 3; k++) blended[o * 3 + k] = (base[o * 3 + k] * wb + acc[o * 3 + k]) / total;
  }

  // Reach into the unseen. A texel no accepted photograph faced (best facing under
  // 0.3) still shows the reconstruction's guess. Real texture a few millimetres away
  // is a better guess than the model's: grow the painted colour into the unseen
  // texels, chart by chart in atlas space, and fade it back toward the guess with
  // distance so a long reach does not smear one colour across a whole unseen side.
  {
    const REACH = Math.round(size / 48);   // ~43 texels at 2048, a few millimetres on a figure
    const dist = new Int16Array(size * size).fill(-1);
    const colour = blended.slice();
    let frontier = [];
    for (let o = 0; o < size * size; o++) {
      if (!covered[o]) continue;
      // a texel only seen at a grazing angle is as good as unseen: its photo weight
      // is tiny and the guess would win, so let the reach fill it from its neighbours
      if (seenBest[o] >= 0.5 ** 8) { dist[o] = 0; frontier.push(o); }   // facing >= 0.5
    }
    let grown = 0;
    for (let d = 1; d <= REACH && frontier.length; d++) {
      const next = [];
      for (const o of frontier) {
        const x = o % size, y = (o - x) / size;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= size || yy >= size) continue;
          const q = yy * size + xx;
          if (!covered[q] || dist[q] >= 0) continue;
          // mean of the already-reached neighbours
          let r = 0, g = 0, b = 0, n = 0;
          for (const [ex, ey] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const px2 = xx + ex, py2 = yy + ey;
            if (px2 < 0 || py2 < 0 || px2 >= size || py2 >= size) continue;
            const p2 = py2 * size + px2;
            if (dist[p2] >= 0 && dist[p2] < d) { r += colour[p2 * 3]; g += colour[p2 * 3 + 1]; b += colour[p2 * 3 + 2]; n++; }
          }
          if (!n) continue;
          const w = 1 - d / REACH;
          colour[q * 3] = (r / n) * w + blended[q * 3] * (1 - w);
          colour[q * 3 + 1] = (g / n) * w + blended[q * 3 + 1] * (1 - w);
          colour[q * 3 + 2] = (b / n) * w + blended[q * 3 + 2] * (1 - w);
          dist[q] = d; next.push(q); grown++;
        }
      }
      frontier = next;
    }
    blended.set(colour);
    log(`reached ${grown.toLocaleString()} unseen texels from painted neighbours`);
  }

  const out = Buffer.alloc(size * size * 3);
  const toSrgb = (c) => 255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
  for (let o = 0; o < size * size; o++) {
    for (let k = 0; k < 3; k++) {
      out[o * 3 + k] = Math.max(0, Math.min(255, Math.round(toSrgb(Math.max(0, Math.min(1, blended[o * 3 + k]))))));
    }
  }
  await writeFile(join(dir, 'photo-splat.jpg'), await readFile(join(dir, 'photo.jpg')));
  await sharp(out, { raw: { width: size, height: size, channels: 3 } })
    .jpeg({ quality: 90, chromaSubsampling: '4:4:4' })
    .toFile(join(dir, 'photo.jpg'));
  return used;
}
