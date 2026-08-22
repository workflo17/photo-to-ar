/**
 * Bake the detail of a dense mesh into a normal map on a coarse one.
 *
 * The shipped Blank is sixty thousand triangles, which is right for a phone in AR and
 * wrong for a figurine: the carved hull of a Shin Godzilla figure carries its scales,
 * its tail ridges and the edges of its dorsal fins in half a million triangles, and
 * simplification planes all of that away. The game-asset answer is the one used here.
 * Keep the coarse mesh, and for every texel of its atlas look up the nearest point on
 * the dense mesh and store that surface's normal, expressed in the coarse triangle's
 * tangent frame. The renderer then shades the coarse mesh as if the detail were there.
 *
 * The source is the RAW splat mesh, before the carve replaces it. The carved hull was
 * tried first and is clean but blunt: at 4 mm voxels it carries a staircase that
 * baked as 20 degrees of speckle and none of the scales. The raw mesh is a double
 * wall full of pockets (see lib/carve.mjs), but the lookup below only accepts
 * triangles facing the same way as the coarse surface, which excludes the ghost sheet
 * (faces inward) and the tunnel walls (edge-on), and what remains is the skin.
 *
 * The tangent frame is derived per triangle from the atlas UVs, the same way
 * mikktspace derives it at export, so the map and the exported tangents agree.
 */

/** Area-weighted vertex normals for an indexed mesh. */
function vertexNormals(positions, indices) {
  const n = positions.length / 3;
  const out = new Float32Array(n * 3);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2];
    const e1 = [positions[b * 3] - positions[a * 3], positions[b * 3 + 1] - positions[a * 3 + 1], positions[b * 3 + 2] - positions[a * 3 + 2]];
    const e2 = [positions[c * 3] - positions[a * 3], positions[c * 3 + 1] - positions[a * 3 + 1], positions[c * 3 + 2] - positions[a * 3 + 2]];
    const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
    for (const v of [a, b, c]) { out[v * 3] += nx; out[v * 3 + 1] += ny; out[v * 3 + 2] += nz; }
  }
  for (let i = 0; i < n; i++) {
    const l = Math.hypot(out[i * 3], out[i * 3 + 1], out[i * 3 + 2]) || 1;
    out[i * 3] /= l; out[i * 3 + 1] /= l; out[i * 3 + 2] /= l;
  }
  return out;
}

/**
 * @param atlas   the cube atlas (unwelded positions, normals, uvs, indices)
 * @param hi      { positions, indices } of the dense mesh, in the Blank's final frame
 * @param size    map resolution
 * @returns { rgb: Buffer, size, coverage, strength } where strength is the mean
 *          angle, in degrees, between the coarse and dense normals: a measure of how
 *          much detail the map actually carries
 */
export function bakeDetail(atlas, hi, size = 1024, { shade = null } = {}) {
  const { positions, normals, uvs, indices, colours: coarseC } = atlas;
  const hiN = vertexNormals(hi.positions, hi.indices);
  const hiC = hi.colours || null;
  // colour output is sRGB-encoded, as a baseColorTexture must be
  const toSrgb = new Uint8ClampedArray(4096);
  for (let i = 0; i < 4096; i++) { const c = i / 4095; toSrgb[i] = 255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055); }
  const enc = (v) => toSrgb[Math.max(0, Math.min(4095, Math.round(v * 4095)))];
  const colour = hiC ? new Uint8ClampedArray(size * size * 3) : null;
  const hn = hi.positions.length / 3;

  // typical dense-mesh spacing sets the search radius
  let edgeSum = 0, edges = 0;
  for (let t = 0; t < hi.indices.length && edges < 20000; t += 3) {
    const a = hi.indices[t], b = hi.indices[t + 1];
    edgeSum += Math.hypot(hi.positions[a * 3] - hi.positions[b * 3], hi.positions[a * 3 + 1] - hi.positions[b * 3 + 1], hi.positions[a * 3 + 2] - hi.positions[b * 3 + 2]);
    edges++;
  }
  const spacing = edgeSum / Math.max(1, edges);
  const radius = spacing * 3;
  const cell = radius;

  // dense triangles hashed by centroid, with their face normals for the side test
  const hiT = hi.indices.length / 3;
  const faceN = new Float32Array(hiT * 3);
  const grid = new Map();
  const keyOf = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  for (let f = 0; f < hiT; f++) {
    const a = hi.indices[f * 3], b = hi.indices[f * 3 + 1], c = hi.indices[f * 3 + 2];
    const P = hi.positions;
    const e1 = [P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]];
    const e2 = [P[c * 3] - P[a * 3], P[c * 3 + 1] - P[a * 3 + 1], P[c * 3 + 2] - P[a * 3 + 2]];
    const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
    const l = Math.hypot(nx, ny, nz) || 1;
    faceN[f * 3] = nx / l; faceN[f * 3 + 1] = ny / l; faceN[f * 3 + 2] = nz / l;
    const k = keyOf((P[a * 3] + P[b * 3] + P[c * 3]) / 3, (P[a * 3 + 1] + P[b * 3 + 1] + P[c * 3 + 1]) / 3, (P[a * 3 + 2] + P[b * 3 + 2] + P[c * 3 + 2]) / 3);
    let l2 = grid.get(k); if (!l2) grid.set(k, l2 = []); l2.push(f);
  }

  // Closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5),
  // returning the squared distance and the barycentric weights of that point.
  const cp = [0, 0, 0];
  const closest = (px, py, pz, a, b, c) => {
    const P = hi.positions;
    const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
    const abx = P[b * 3] - ax, aby = P[b * 3 + 1] - ay, abz = P[b * 3 + 2] - az;
    const acx = P[c * 3] - ax, acy = P[c * 3 + 1] - ay, acz = P[c * 3 + 2] - az;
    const apx = px - ax, apy = py - ay, apz = pz - az;
    const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
    if (d1 <= 0 && d2 <= 0) { cp[0] = 1; cp[1] = 0; cp[2] = 0; return apx * apx + apy * apy + apz * apz; }
    const bpx = px - P[b * 3], bpy = py - P[b * 3 + 1], bpz = pz - P[b * 3 + 2];
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) { cp[0] = 0; cp[1] = 1; cp[2] = 0; return bpx * bpx + bpy * bpy + bpz * bpz; }
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const v = d1 / (d1 - d3);
      cp[0] = 1 - v; cp[1] = v; cp[2] = 0;
      const qx = ax + abx * v - px, qy = ay + aby * v - py, qz = az + abz * v - pz;
      return qx * qx + qy * qy + qz * qz;
    }
    const cpx = px - P[c * 3], cpy = py - P[c * 3 + 1], cpz = pz - P[c * 3 + 2];
    const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0 && d5 <= d6) { cp[0] = 0; cp[1] = 0; cp[2] = 1; return cpx * cpx + cpy * cpy + cpz * cpz; }
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const w = d2 / (d2 - d6);
      cp[0] = 1 - w; cp[1] = 0; cp[2] = w;
      const qx = ax + acx * w - px, qy = ay + acy * w - py, qz = az + acz * w - pz;
      return qx * qx + qy * qy + qz * qz;
    }
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
      const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      cp[0] = 0; cp[1] = 1 - w; cp[2] = w;
      const qx = P[b * 3] + (P[c * 3] - P[b * 3]) * w - px, qy = P[b * 3 + 1] + (P[c * 3 + 1] - P[b * 3 + 1]) * w - py, qz = P[b * 3 + 2] + (P[c * 3 + 2] - P[b * 3 + 2]) * w - pz;
      return qx * qx + qy * qy + qz * qz;
    }
    const denom = 1 / (va + vb + vc);
    const v = vb * denom, w = vc * denom;
    cp[0] = 1 - v - w; cp[1] = v; cp[2] = w;
    const qx = ax + abx * v + acx * w - px, qy = ay + aby * v + acy * w - py, qz = az + abz * v + acz * w - pz;
    return qx * qx + qy * qy + qz * qz;
  };

  // The normal at the closest point of the dense surface, interpolated from that
  // triangle's vertex normals. Only triangles facing the same way as the coarse
  // surface are candidates: the search reaches through a tail or a fin, which is a
  // few voxels thick, and the far side's normal is not the detail wanted. (A blend of
  // nearby vertex normals was tried first and left a self-bake of the coarse mesh
  // against itself at 10 to 24 degrees of relief, when it should be flat.)
  const sample = (x, y, z, nx, ny, nz, out) => {
    const gx = Math.floor(x / cell), gy = Math.floor(y / cell), gz = Math.floor(z / cell);
    let bestD = Infinity, bestF = -1, b0 = 0, b1 = 0, b2 = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const l = grid.get(`${gx + dx},${gy + dy},${gz + dz}`); if (!l) continue;
      for (const f of l) {
        if (faceN[f * 3] * nx + faceN[f * 3 + 1] * ny + faceN[f * 3 + 2] * nz < 0.2) continue;
        const d2 = closest(x, y, z, hi.indices[f * 3], hi.indices[f * 3 + 1], hi.indices[f * 3 + 2]);
        if (d2 < bestD) { bestD = d2; bestF = f; b0 = cp[0]; b1 = cp[1]; b2 = cp[2]; }
      }
    }
    if (bestF < 0) return false;
    const a = hi.indices[bestF * 3], b = hi.indices[bestF * 3 + 1], c = hi.indices[bestF * 3 + 2];
    let sx = b0 * hiN[a * 3] + b1 * hiN[b * 3] + b2 * hiN[c * 3];
    let sy = b0 * hiN[a * 3 + 1] + b1 * hiN[b * 3 + 1] + b2 * hiN[c * 3 + 1];
    let sz = b0 * hiN[a * 3 + 2] + b1 * hiN[b * 3 + 2] + b2 * hiN[c * 3 + 2];
    const l = Math.hypot(sx, sy, sz) || 1;
    out[0] = sx / l; out[1] = sy / l; out[2] = sz / l;
    if (hiC) {
      for (let k = 0; k < 3; k++) out[3 + k] = b0 * hiC[a * 3 + k] + b1 * hiC[b * 3 + k] + b2 * hiC[c * 3 + k];
    }
    return true;
  };

  const rgb = new Uint8ClampedArray(size * size * 3);
  const filled = new Uint8Array(size * size);
  const px = new Float32Array(uvs.length / 2), py = new Float32Array(uvs.length / 2);
  for (let i = 0; i < uvs.length / 2; i++) {
    px[i] = uvs[i * 2] * (size - 1);
    py[i] = uvs[i * 2 + 1] * (size - 1);   // glTF convention, see atlas.mjs
  }

  const hiNrm = [0, 0, 0, 0, 0, 0];
  let angleSum = 0, angleCount = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
    const ax = px[i0], ay = py[i0], bx = px[i1], by = py[i1], cx = px[i2], cy = py[i2];
    const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if (Math.abs(area) < 1e-9) continue;

    // Tangent frame from the UV gradient, per triangle. This is what mikktspace
    // computes at export before its per-vertex averaging, and with an unwelded atlas
    // every vertex belongs to one triangle, so the two agree.
    const P0 = [positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]];
    const E1 = [positions[i1 * 3] - P0[0], positions[i1 * 3 + 1] - P0[1], positions[i1 * 3 + 2] - P0[2]];
    const E2 = [positions[i2 * 3] - P0[0], positions[i2 * 3 + 1] - P0[1], positions[i2 * 3 + 2] - P0[2]];
    const du1 = uvs[i1 * 2] - uvs[i0 * 2], dv1 = uvs[i1 * 2 + 1] - uvs[i0 * 2 + 1];
    const du2 = uvs[i2 * 2] - uvs[i0 * 2], dv2 = uvs[i2 * 2 + 1] - uvs[i0 * 2 + 1];
    const det = du1 * dv2 - du2 * dv1;
    if (Math.abs(det) < 1e-12) continue;
    const r = 1 / det;
    const T = [(E1[0] * dv2 - E2[0] * dv1) * r, (E1[1] * dv2 - E2[1] * dv1) * r, (E1[2] * dv2 - E2[2] * dv1) * r];
    const Bref = [(E2[0] * du1 - E1[0] * du2) * r, (E2[1] * du1 - E1[1] * du2) * r, (E2[2] * du1 - E1[2] * du2) * r];

    const lo = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const hiX = Math.min(size - 1, Math.ceil(Math.max(ax, bx, cx)));
    const vlo = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const vhi = Math.min(size - 1, Math.ceil(Math.max(ay, by, cy)));

    for (let y = vlo; y <= vhi; y++) {
      for (let x = lo; x <= hiX; x++) {
        const w0 = ((bx - ax) * (y + 0.5 - ay) - (x + 0.5 - ax) * (by - ay)) / area;
        const w1 = ((cx - bx) * (y + 0.5 - by) - (x + 0.5 - bx) * (cy - by)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < -0.002 || w1 < -0.002 || w2 < -0.002) continue;
        // weights are opposite-vertex: w1 -> i0, w2 -> i1, w0 -> i2
        const wx = w1 * positions[i0 * 3] + w2 * positions[i1 * 3] + w0 * positions[i2 * 3];
        const wy = w1 * positions[i0 * 3 + 1] + w2 * positions[i1 * 3 + 1] + w0 * positions[i2 * 3 + 1];
        const wz = w1 * positions[i0 * 3 + 2] + w2 * positions[i1 * 3 + 2] + w0 * positions[i2 * 3 + 2];
        let nx = w1 * normals[i0 * 3] + w2 * normals[i1 * 3] + w0 * normals[i2 * 3];
        let ny = w1 * normals[i0 * 3 + 1] + w2 * normals[i1 * 3 + 1] + w0 * normals[i2 * 3 + 1];
        let nz = w1 * normals[i0 * 3 + 2] + w2 * normals[i1 * 3 + 2] + w0 * normals[i2 * 3 + 2];
        const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;

        if (!sample(wx, wy, wz, nx, ny, nz, hiNrm)) {
          // Nothing of the dense mesh within reach, which happens where the carved
          // surface smoothed over a pocketed region. Flat relief, and the coarse
          // mesh's own colour: a texel left black here shows as a black patch.
          if (colour && coarseC) {
            // no shade factor here: the coarse colours were de-lit already
            const o = (y * size + x) * 3;
            for (let k = 0; k < 3; k++) {
              const c = w1 * coarseC[i0 * 3 + k] + w2 * coarseC[i1 * 3 + k] + w0 * coarseC[i2 * 3 + k];
              colour[o + k] = enc(Math.min(1, Math.max(0, c)));
            }
            rgb[o] = 128; rgb[o + 1] = 128; rgb[o + 2] = 255;
            filled[y * size + x] = 1;
          }
          continue;
        }

        // Gram-Schmidt the tangent against the interpolated normal, then a bitangent
        // whose handedness follows the UV layout
        const tn = T[0] * nx + T[1] * ny + T[2] * nz;
        let tx = T[0] - nx * tn, ty = T[1] - ny * tn, tz = T[2] - nz * tn;
        const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
        let bx2 = ny * tz - nz * ty, by2 = nz * tx - nx * tz, bz2 = nx * ty - ny * tx;
        if (bx2 * Bref[0] + by2 * Bref[1] + bz2 * Bref[2] < 0) { bx2 = -bx2; by2 = -by2; bz2 = -bz2; }

        let lx = hiNrm[0] * tx + hiNrm[1] * ty + hiNrm[2] * tz;
        let ly = hiNrm[0] * bx2 + hiNrm[1] * by2 + hiNrm[2] * bz2;
        let lz = hiNrm[0] * nx + hiNrm[1] * ny + hiNrm[2] * nz;
        // a dense normal facing away from the coarse surface is a lookup that landed on
        // the far side of a thin part; fold it back rather than bake a hole
        if (lz < 0.05) { lz = 0.05; const ll = Math.hypot(lx, ly, lz); lx /= ll; ly /= ll; lz /= ll; }
        angleSum += Math.acos(Math.min(1, lz)); angleCount++;

        const o = (y * size + x) * 3;
        rgb[o] = (lx * 0.5 + 0.5) * 255;
        rgb[o + 1] = (ly * 0.5 + 0.5) * 255;
        rgb[o + 2] = (lz * 0.5 + 0.5) * 255;
        if (colour) {
          const f = shade ? shade(nx, ny, nz) : 1;
          for (let k = 0; k < 3; k++) colour[o + k] = enc(Math.min(1, Math.max(0, hiNrm[3 + k] * f)));
        }
        filled[y * size + x] = 1;
      }
    }
  }

  // A small median over the colour, filled texels only. The dense mesh's colour is
  // also its noise: on the reference chair the raw splat colours carry white flecks a
  // few texels wide, which the coarse bake used to average away and which came back
  // as a shattered surface. A 5x5 median removes them and keeps anything wider, which
  // is every feature that matters (an eye on a figurine is twenty texels).
  if (colour) {
    const R = 2;
    const src = colour.slice();
    const win = new Uint8ClampedArray((2 * R + 1) * (2 * R + 1));
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (!filled[y * size + x]) continue;
        for (let k = 0; k < 3; k++) {
          let n = 0;
          for (let dy = -R; dy <= R; dy++) {
            const yy = y + dy; if (yy < 0 || yy >= size) continue;
            for (let dx = -R; dx <= R; dx++) {
              const xx = x + dx; if (xx < 0 || xx >= size || !filled[yy * size + xx]) continue;
              win[n++] = src[(yy * size + xx) * 3 + k];
            }
          }
          if (n < 5) continue;
          const sorted = win.subarray(0, n).slice().sort();
          colour[(y * size + x) * 3 + k] = sorted[n >> 1];
        }
      }
    }
  }

  // dilate, as the colour bake does, so bilinear reads past a chart edge stay sane
  for (let pass = 0; pass < 6; pass++) {
    const next = filled.slice();
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (filled[y * size + x]) continue;
        let r = 0, g = 0, b = 0, cr = 0, cg = 0, cb = 0, n = 0;
        for (const [dy, dx] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= size || xx < 0 || xx >= size || !filled[yy * size + xx]) continue;
          const o = (yy * size + xx) * 3;
          r += rgb[o]; g += rgb[o + 1]; b += rgb[o + 2]; n++;
          if (colour) { cr += colour[o]; cg += colour[o + 1]; cb += colour[o + 2]; }
        }
        if (!n) continue;
        const o = (y * size + x) * 3;
        rgb[o] = r / n; rgb[o + 1] = g / n; rgb[o + 2] = b / n;
        if (colour) { colour[o] = cr / n; colour[o + 1] = cg / n; colour[o + 2] = cb / n; }
        next[y * size + x] = 1;
      }
    }
    filled.set(next);
  }
  // unfilled texels are flat
  for (let i = 0; i < size * size; i++) {
    if (filled[i]) continue;
    rgb[i * 3] = 128; rgb[i * 3 + 1] = 128; rgb[i * 3 + 2] = 255;
  }

  const coverage = filled.reduce((s, v) => s + v, 0) / filled.length;
  const strength = angleCount ? (angleSum / angleCount) * 180 / Math.PI : 0;
  return {
    rgb: Buffer.from(rgb.buffer, rgb.byteOffset, rgb.length),
    colour: colour ? Buffer.from(colour) : null,
    size, coverage, strength,
  };
}
