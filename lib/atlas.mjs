/**
 * A unique UV layout, and a baker that rasterises per-vertex colour into it.
 *
 * The Blank's usual UVs are box-projected in metres so that a weave tiles at a real
 * scale. That is exactly right for a repeating fabric and exactly wrong for a
 * photograph-derived colour, because tiling UVs are not unique: many points on the
 * surface land on the same texel, so there is nowhere to put a per-point colour.
 *
 * So the coloured path gets its own layout: a six-tile cube atlas. Each triangle picks
 * the dominant SIGNED axis of its face normal and is projected onto that axis plane,
 * into one of six tiles in a 3x2 grid. Six signed axes rather than three unsigned ones
 * is what makes it unique: an unsigned box projection maps the front and the back of an
 * object to the same place, which is the same overlap problem in a smaller disguise.
 *
 * The choice is per FACE, not per vertex. Per vertex, a triangle whose corners disagree
 * about the dominant axis gets stretched across two tiles and smears the whole atlas.
 *
 * This is not as efficient as a real unwrap. It has six charts where xatlas would find
 * a dozen well-packed ones, and it distorts on surfaces that face diagonally. It needs
 * no dependency, runs in a second, and is deterministic, which for a first coloured
 * export is the right trade.
 */

const AXES = [
  // [axis, sign, uAxis, vAxis] with the tile index matching the array order
  [0, +1, 2, 1], [0, -1, 2, 1],
  [1, +1, 0, 2], [1, -1, 0, 2],
  [2, +1, 0, 1], [2, -1, 0, 1],
];
const COLS = 3, ROWS = 2;

/**
 * Build the atlas.
 *
 * Returns unwelded arrays: three vertices per triangle, because a vertex on the seam
 * between two tiles needs a different UV in each and cannot be shared.
 */
export function cubeAtlas(positions, normals, colours, indices) {
  const triCount = indices.length / 3;
  const pos = new Float32Array(triCount * 9);
  const nrm = new Float32Array(triCount * 9);
  const col = colours ? new Float32Array(triCount * 9) : null;
  const uv = new Float32Array(triCount * 6);

  // bounds, so each tile's projection fills its cell
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (positions[i + k] < mn[k]) mn[k] = positions[i + k];
      if (positions[i + k] > mx[k]) mx[k] = positions[i + k];
    }
  }
  const size = mx.map((v, i) => Math.max(v - mn[i], 1e-6));

  // a margin inside each tile, so the dilation pass has somewhere to bleed into and
  // bilinear filtering at a tile edge cannot reach into its neighbour
  const M = 0.02;

  // Pass one: each face's own choice, the signed axis its normal leans on most.
  const faceN = new Float32Array(triCount * 3);
  const tileOf = new Uint8Array(triCount);
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
    for (let k = 0; k < 3; k++) {
      a[k] = positions[i0 * 3 + k]; b[k] = positions[i1 * 3 + k]; c[k] = positions[i2 * 3 + k];
    }
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const f = [e1[1] * e2[2] - e1[2] * e2[1],
               e1[2] * e2[0] - e1[0] * e2[2],
               e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(f[0], f[1], f[2]) || 1;
    faceN[t * 3] = f[0] / l; faceN[t * 3 + 1] = f[1] / l; faceN[t * 3 + 2] = f[2] / l;
    let tile = 0, best = -Infinity;
    for (let s = 0; s < 6; s++) {
      const [axis, sign] = AXES[s];
      const d = f[axis] * sign;
      if (d > best) { best = d; tile = s; }
    }
    tileOf[t] = tile;
  }

  // Pass two: follow the neighbours. A face whose normal sits near 45 degrees between
  // two axes picks a tile on its own, and a lone face in a tile is a one-triangle
  // chart: it gets its own dilation ring, its own median neighbourhood, and on a
  // surface painted from several photographs it shows as a small square of the wrong
  // shade. If two of a face's three edge-neighbours agree on another tile and the
  // face still faces that tile's axis at all, it joins them. Three rounds settle it.
  {
    const edgeFace = new Map();
    const key = (u, v) => (u < v ? u * 4294967296 + v : v * 4294967296 + u);
    const nbr = Array.from({ length: triCount }, () => []);
    for (let t = 0; t < triCount; t++) {
      for (let k = 0; k < 3; k++) {
        const kk = key(indices[t * 3 + k], indices[t * 3 + (k + 1) % 3]);
        const other = edgeFace.get(kk);
        if (other === undefined) edgeFace.set(kk, t);
        else { nbr[t].push(other); nbr[other].push(t); }
      }
    }
    let moved = 0;
    const rounds = process.env.ATLAS_ROUNDS !== undefined ? Number(process.env.ATLAS_ROUNDS) : 3;
    for (let round = 0; round < rounds; round++) {
      const next = Uint8Array.from(tileOf);
      for (let t = 0; t < triCount; t++) {
        const votes = [0, 0, 0, 0, 0, 0];
        for (const o of nbr[t]) votes[tileOf[o]]++;
        let bestTile = tileOf[t], bestVotes = votes[tileOf[t]];
        for (let s = 0; s < 6; s++) if (votes[s] > bestVotes) { bestVotes = votes[s]; bestTile = s; }
        if (bestTile === tileOf[t] || bestVotes < 2) continue;
        const [axis, sign] = AXES[bestTile];
        if (faceN[t * 3 + axis] * sign < 0.2) continue;   // would be seen edge-on there
        next[t] = bestTile; moved++;
      }
      tileOf.set(next);
    }
    // Pass three: absorb the charts that are still tiny. A face edge-on to its
    // neighbours' tile is stretched there, but a stretched face in a big chart is
    // invisible and a crisp face alone in its own chart is a fleck. Charts under six
    // faces join the tile most of their outside neighbours use, if the face faces it
    // at all.
    {
      const parent = new Int32Array(triCount);
      for (let t = 0; t < triCount; t++) parent[t] = t;
      const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
      for (let t = 0; t < triCount; t++) for (const o of nbr[t]) if (tileOf[o] === tileOf[t]) { const a = find(t), b = find(o); if (a !== b) parent[a] = b; }
      const members = new Map();
      for (let t = 0; t < triCount; t++) { const r = find(t); let l = members.get(r); if (!l) members.set(r, l = []); l.push(t); }
      let absorbed = 0;
      for (const list of members.values()) {
        if (list.length >= 6) continue;
        const votes = [0, 0, 0, 0, 0, 0];
        for (const t of list) for (const o of nbr[t]) if (tileOf[o] !== tileOf[t]) votes[tileOf[o]]++;
        let bestTile = -1, bestVotes = 0;
        for (let s = 0; s < 6; s++) if (votes[s] > bestVotes) { bestVotes = votes[s]; bestTile = s; }
        if (bestTile < 0) continue;
        const [axis, sign] = AXES[bestTile];
        for (const t of list) { if (faceN[t * 3 + axis] * sign > 0.02) { tileOf[t] = bestTile; absorbed++; } }
      }
      cubeAtlas.lastAbsorbed = absorbed;
    }
    cubeAtlas.lastMoved = moved;
  }

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
    const tile = tileOf[t];
    const [, , uAxis, vAxis] = AXES[tile];
    const tx = (tile % COLS) / COLS, ty = Math.floor(tile / COLS) / ROWS;

    for (const [n, src] of [[0, i0], [1, i1], [2, i2]]) {
      const o = (t * 3 + n);
      for (let k = 0; k < 3; k++) {
        pos[o * 3 + k] = positions[src * 3 + k];
        nrm[o * 3 + k] = normals ? normals[src * 3 + k] : 0;
        if (col) col[o * 3 + k] = colours[src * 3 + k];
      }
      const u = (positions[src * 3 + uAxis] - mn[uAxis]) / size[uAxis];
      const v = (positions[src * 3 + vAxis] - mn[vAxis]) / size[vAxis];
      uv[o * 2] = tx + (M + u * (1 - 2 * M)) / COLS;
      uv[o * 2 + 1] = ty + (M + v * (1 - 2 * M)) / ROWS;
    }
  }

  const idx = new Uint32Array(triCount * 3);
  for (let i = 0; i < idx.length; i++) idx[i] = i;
  return { positions: pos, normals: nrm, colours: col, uvs: uv, indices: idx };
}

/**
 * Rasterise per-vertex colour into an RGB image through the atlas UVs.
 *
 * Flat scanline fill with barycentric interpolation, then a dilation pass. The dilation
 * is not optional: a texture sampled with bilinear filtering reads past the edge of
 * every chart, and without bled-out colour there the result is a dark fringe along
 * every seam on the model.
 */
export function bakeColours({ uvs, colours, indices }, size = 1024) {
  const rgb = new Uint8ClampedArray(size * size * 3);
  const filled = new Uint8Array(size * size);

  // glTF COLOR_0 is LINEAR; a baseColorTexture is sRGB-encoded. Writing linear values
  // straight into the image skips the transfer function and crushes everything dark:
  // on the reference chair a linear 0.036 became a near-black texel instead of the
  // 0.20 mid-shadow it represents, which read as speckle scattered through the cream.
  // A 256-entry table because this runs per texel per channel.
  const toSrgb = new Uint8ClampedArray(4096);
  for (let i = 0; i < 4096; i++) {
    const c = i / 4095;
    toSrgb[i] = 255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
  }
  const enc = (v) => toSrgb[Math.max(0, Math.min(4095, Math.round(v * 4095)))];

  const px = new Float32Array(uvs.length / 2);
  const py = new Float32Array(uvs.length / 2);
  for (let i = 0; i < uvs.length / 2; i++) {
    px[i] = uvs[i * 2] * (size - 1);
    // glTF's texture origin is the TOP-left: v grows downward, so row = v. This was
    // written as (1 - v) for a long time, the OpenGL convention, and every glTF viewer
    // read the atlas upside down: with six tiles that lands the OPPOSITE side's chart
    // on each face, mirrored, and a chair wore its legs across the top of its back.
    // The studio's TextureLoader and the USDZ were reading it the other way and
    // looked right, which is how it hid.
    py[i] = uvs[i * 2 + 1] * (size - 1);
  }

  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
    const ax = px[i0], ay = py[i0], bx = px[i1], by = py[i1], cx = px[i2], cy = py[i2];
    const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if (Math.abs(area) < 1e-9) continue;

    const lo = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const hi = Math.min(size - 1, Math.ceil(Math.max(ax, bx, cx)));
    const vlo = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const vhi = Math.min(size - 1, Math.ceil(Math.max(ay, by, cy)));

    for (let y = vlo; y <= vhi; y++) {
      for (let x = lo; x <= hi; x++) {
        const w0 = ((bx - ax) * (y + 0.5 - ay) - (x + 0.5 - ax) * (by - ay)) / area;
        const w1 = ((cx - bx) * (y + 0.5 - by) - (x + 0.5 - bx) * (cy - by)) / area;
        const w2 = 1 - w0 - w1;
        // one sign test, not two: the winding is known because the atlas built it
        if (w0 < -0.002 || w1 < -0.002 || w2 < -0.002) continue;
        // Barycentric bookkeeping, which is easy to get wrong and looks like noise
        // rather than like an error when you do: an edge function evaluated at a point
        // is proportional to the distance from that edge, so it is the weight of the
        // vertex OPPOSITE it. w0 is edge AB, so it weights C. w1 is edge BC, so it
        // weights A. w2 is the remainder, so it weights B. Rotating these by one
        // scatters each triangle's darkest corner colour into its lightest neighbour
        // and speckles the whole atlas.
        const o = (y * size + x) * 3;
        for (let k = 0; k < 3; k++) {
          rgb[o + k] = enc(w1 * colours[i0 * 3 + k] + w2 * colours[i1 * 3 + k] + w0 * colours[i2 * 3 + k]);
        }
        filled[y * size + x] = 1;
      }
    }
  }

  // Dilate. Each pass pushes colour one texel outward from whatever is already filled.
  for (let pass = 0; pass < 6; pass++) {
    const next = filled.slice();
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (filled[y * size + x]) continue;
        let r = 0, g = 0, bl = 0, n = 0;
        for (const [dy, dx] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= size || nx < 0 || nx >= size) continue;
          if (!filled[ny * size + nx]) continue;
          const o = (ny * size + nx) * 3;
          r += rgb[o]; g += rgb[o + 1]; bl += rgb[o + 2]; n++;
        }
        if (!n) continue;
        const o = (y * size + x) * 3;
        rgb[o] = r / n; rgb[o + 1] = g / n; rgb[o + 2] = bl / n;
        next[y * size + x] = 1;
      }
    }
    filled.set(next);
  }

  const coverage = filled.reduce((s, v) => s + v, 0) / filled.length;
  return { rgb: Buffer.from(rgb.buffer, rgb.byteOffset, rgb.length), size, coverage };
}
