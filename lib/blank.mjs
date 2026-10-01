/**
 * Raw reconstruction in, a Blank out.
 *
 * A Blank is the reusable, untextured product asset: floor removed, simplified to a
 * phone-sized triangle budget, sitting upright with its feet on y=0 at true size,
 * carrying box-projected UVs and a segmentation into regions you can paint
 * separately. It has no material. Materials belong to a Look, and one Blank can
 * wear many Looks without being rebuilt.
 *
 * The interface is one function because every decision a caller makes is a number:
 * how tall is the real thing, how many triangles can the target afford, how coarse
 * should the regions be. Behind it sits the entire difference between a marching
 * cubes blob and something a shopper can stand in a room.
 *
 * The floor cut and the box-projection are ported from ~/product-viz/tools/finish_3d.mjs,
 * where they were written and proven. The segmentation is new: that file assumed one
 * material for the whole piece.
 */
import { carveSkin } from './carve.mjs';
import { bakeDetail } from './detail.mjs';
import { silhouette, iou, viewFor } from './align.mjs';
import { NodeIO } from '@gltf-transform/core';
import { simplify, weld } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
import { writeFile } from 'node:fs/promises';
import sharp from 'sharp';
import { cubeAtlas, bakeColours } from './atlas.mjs';

/* -------------------------------------------------------------- the floor slab */
/**
 * Single-image reconstruction cannot tell the product from the set it was shot on,
 * so a studio seamless and its contact shadow come back as a slab of geometry under
 * the piece. The slab is unmistakable by width rather than by vertex count: it spans
 * the whole frame while the legs standing on it span a few centimetres. Find the
 * lowest band that is anomalously wide and cut above it.
 */
function dropGroundPlane(prim, log) {
  const pos = prim.getAttribute('POSITION');
  const idx = prim.getIndices();
  const n = pos.getCount();
  const mn = [], mx = [];
  pos.getMin(mn); pos.getMax(mx);

  const BINS = 60;
  const lo = new Float64Array(BINS).fill(Infinity);
  const hi = new Float64Array(BINS).fill(-Infinity);
  const count = new Int32Array(BINS);
  const v = [0, 0, 0];
  const binOf = (y) => Math.min(BINS - 1, Math.max(0, Math.floor(((y - mn[1]) / (mx[1] - mn[1])) * BINS)));
  for (let i = 0; i < n; i++) {
    pos.getElement(i, v);
    const b = binOf(v[1]);
    lo[b] = Math.min(lo[b], v[0]);
    hi[b] = Math.max(hi[b], v[0]);
    count[b]++;
  }
  const span = Array.from({ length: BINS }, (_, b) => (hi[b] > lo[b] ? hi[b] - lo[b] : 0));

  // Two signals, because neither one catches both kinds of floor.
  //
  // Width: a plate is wider than the band immediately above it, because what stands
  // on it are the legs. Comparing against the whole body fails, since the widest part
  // of a chair is nearly as wide as its shadow.
  //
  // Density: once the photograph has been matted, what survives is not a full
  // seamless but the contact shadow, a thin sheet that is barely wider than the
  // footprint and would slip under any width threshold loose enough to be safe. It
  // gives itself away by vertex count instead: on the reference chair the bottom bin
  // held 6,236 vertices against 296 in the band above, a 21x spike, while being only
  // 1.31x wider. A width-only test missed it and left a plate under the model.
  let cutBin = -1;
  const searchTo = Math.floor(BINS * 0.15);
  for (let b = 0; b < searchTo; b++) {
    const aboveSpan = span[b + 1] || span[b];
    // median density of the bands just above, which is robust to one sparse bin
    const near = [...count.slice(b + 1, b + 6)].filter((c) => c > 0).sort((x, y) => x - y);
    const aboveCount = near.length ? near[near.length >> 1] : 0;
    const wide = aboveSpan > 0 && span[b] > aboveSpan * 1.25;
    const dense = aboveCount > 0 && count[b] > aboveCount * 6;
    if (wide && dense) cutBin = b;             // both: certain
    else if (wide && span[b] > aboveSpan * 1.4) cutBin = b;   // width alone, but strongly
    else if (dense && count[b] > aboveCount * 12) cutBin = b; // density alone, but strongly
  }
  if (cutBin < 0) { log('no ground plane detected'); return 0; }

  const cutY = mn[1] + ((cutBin + 1) / BINS) * (mx[1] - mn[1]);
  const tri = idx.getArray();
  const kept = [];
  const a = [0, 0, 0], b2 = [0, 0, 0], c = [0, 0, 0];
  for (let t = 0; t < tri.length; t += 3) {
    pos.getElement(tri[t], a);
    pos.getElement(tri[t + 1], b2);
    pos.getElement(tri[t + 2], c);
    // keep a triangle when its centroid sits above the cut, which leaves a clean
    // edge rather than the fringe you get from testing individual vertices
    if ((a[1] + b2[1] + c[1]) / 3 > cutY) kept.push(tri[t], tri[t + 1], tri[t + 2]);
  }
  const removed = tri.length / 3 - kept.length / 3;
  idx.setArray(new Uint32Array(kept));
  log(`floor cut at y=${cutY.toFixed(3)}, removed ${removed.toLocaleString()} triangles`);
  return removed;
}

/* ------------------------------------------------------------------- normals */
function computeNormals(doc, prim) {
  const pos = prim.getAttribute('POSITION');
  const idx = prim.getIndices();
  const n = pos.getCount();
  const acc = new Float32Array(n * 3);
  const tri = idx ? idx.getArray() : null;
  const triCount = tri ? tri.length / 3 : n / 3;
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
  for (let t = 0; t < triCount; t++) {
    const i0 = tri ? tri[t * 3] : t * 3;
    const i1 = tri ? tri[t * 3 + 1] : t * 3 + 1;
    const i2 = tri ? tri[t * 3 + 2] : t * 3 + 2;
    pos.getElement(i0, a); pos.getElement(i1, b); pos.getElement(i2, c);
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    // cross product magnitude is twice the triangle area, so no normalising here:
    // bigger triangles should pull harder on the vertex normal
    const f = [e1[1] * e2[2] - e1[2] * e2[1],
               e1[2] * e2[0] - e1[0] * e2[2],
               e1[0] * e2[1] - e1[1] * e2[0]];
    for (const i of [i0, i1, i2]) {
      acc[i * 3] += f[0]; acc[i * 3 + 1] += f[1]; acc[i * 3 + 2] += f[2];
    }
  }
  for (let i = 0; i < n; i++) {
    const l = Math.hypot(acc[i * 3], acc[i * 3 + 1], acc[i * 3 + 2]) || 1;
    acc[i * 3] /= l; acc[i * 3 + 1] /= l; acc[i * 3 + 2] /= l;
  }
  prim.setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(acc));
}

/* ----------------------------------------------------------------- duplicates */
/**
 * Drop triangles that already exist, in either winding.
 *
 * The splat isosurface carries 8.8% duplicate faces after simplification, and most of
 * them are an inside-out twin of a face that is already present: the same three
 * vertices wound the other way, coincident in space. Those twins are what the specks
 * are. Two coincident faces with opposite normals z-fight, so the renderer flickers
 * between lit and unlit on every frame and the result reads as dark flecks scattered
 * over the surface. They also defeat orientation repair outright, since a face and
 * its twin can never be made to agree.
 *
 * Keep the first occurrence. Orientation repair runs next and decides which way it
 * should wind, so which copy survives does not matter.
 */
function dedupeFaces(prim, log) {
  const idxAcc = prim.getIndices();
  const tri = idxAcc.getArray();
  const seen = new Set();
  const kept = [];
  let dropped = 0;
  for (let t = 0; t < tri.length; t += 3) {
    const a = tri[t], b = tri[t + 1], c = tri[t + 2];
    if (a === b || b === c || a === c) { dropped++; continue; }   // degenerate
    const lo = Math.min(a, b, c), hi = Math.max(a, b, c), mid = a + b + c - lo - hi;
    const k = lo * 4294967296 + mid * 65536 + hi;   // unique while indices stay under 2^16... see below
    // vertex counts here are tens of thousands, so pack into a string when they might
    // not fit three to a double
    const key = hi < 65536 ? k : `${lo},${mid},${hi}`;
    if (seen.has(key)) { dropped++; continue; }
    seen.add(key);
    kept.push(a, b, c);
  }
  idxAcc.setArray(new Uint32Array(kept));
  log(`dropped ${dropped.toLocaleString()} duplicate or degenerate faces (${(100 * dropped / (tri.length / 3)).toFixed(1)}%)`);
}

/* ------------------------------------------------------------------ detached */
function dropDetached(prim, log) {
  const idxAcc = prim.getIndices();
  const tri = idxAcc.getArray();
  const pos = prim.getAttribute('POSITION').getArray();
  const n = pos.length / 3;
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (let t = 0; t < tri.length; t += 3) { const a = find(tri[t]), b = find(tri[t + 1]), c = find(tri[t + 2]); parent[a] = b; parent[find(b)] = find(c); }
  const faces = new Map();
  for (let t = 0; t < tri.length; t += 3) { const r = find(tri[t]); faces.set(r, (faces.get(r) || 0) + 1); }
  if (faces.size < 2) return;
  let mainRoot = -1, mainFaces = 0;
  for (const [r, c] of faces) if (c > mainFaces) { mainFaces = c; mainRoot = r; }
  // the main body's box
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) { if (find(i) !== mainRoot) continue; for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], pos[i * 3 + k]); max[k] = Math.max(max[k], pos[i * 3 + k]); } }
  const height = max[1] - min[1];
  // Distance to the main body's SURFACE, not its box: a raised tail tip puts the
  // whole box above a slab floating over the head, and the slab would count as
  // touching. A piece whose nearest main-body vertex is further than 2.5% of the
  // height away from every one of its own vertices is floating.
  const gap = height * 0.025;
  const cell = gap;
  const grid = new Map();
  const keyOf = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  for (let i = 0; i < n; i++) {
    if (find(i) !== mainRoot) continue;
    const k = keyOf(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    let l = grid.get(k); if (!l) grid.set(k, l = []); l.push(i);
  }
  const nearMain = (x, y, z) => {
    const gx = Math.floor(x / cell), gy = Math.floor(y / cell), gz = Math.floor(z / cell);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const l = grid.get(`${gx + dx},${gy + dy},${gz + dz}`); if (!l) continue;
      for (const j of l) if ((pos[j * 3] - x) ** 2 + (pos[j * 3 + 1] - y) ** 2 + (pos[j * 3 + 2] - z) ** 2 < gap * gap) return true;
    }
    return false;
  };
  const drop = new Set();
  for (const [r, c] of faces) {
    if (r === mainRoot || c > mainFaces * 0.15) continue;
    let touches = false;
    for (let i = 0; i < n && !touches; i++) {
      if (find(i) !== r) continue;
      touches = nearMain(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    if (!touches) drop.add(r);
  }
  if (!drop.size) return;
  const kept = [];
  let dropped = 0;
  for (let t = 0; t < tri.length; t += 3) {
    if (drop.has(find(tri[t]))) { dropped++; continue; }
    kept.push(tri[t], tri[t + 1], tri[t + 2]);
  }
  idxAcc.setArray(new Uint32Array(kept));
  log(`dropped ${drop.size} detached piece${drop.size === 1 ? '' : 's'} outside the object (${dropped.toLocaleString()} faces)`);
}

/* ------------------------------------------------------------------ fragments */
/**
 * Drop fragments too small to be part of the product.
 *
 * Once the inside-out twins are removed, what was the inner wall of a double-walled
 * sliver is left behind as tiny floating pieces, a few faces each, sitting just off
 * the surface and catching the light as flakes. Measured on the reference chair: one
 * real component of 51,293 faces and 44 fragments, 42 of them under fifty faces,
 * 1.1% of the mesh in total. Nothing that small is a leg or a cushion.
 */
function dropFragments(prim, log, minFaces = 30) {
  const idxAcc = prim.getIndices();
  const tri = idxAcc.getArray();
  const n = prim.getAttribute('POSITION').getCount();
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const unite = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
  for (let t = 0; t < tri.length; t += 3) { unite(tri[t], tri[t + 1]); unite(tri[t + 1], tri[t + 2]); }
  const size = new Map();
  for (let t = 0; t < tri.length; t += 3) { const r = find(tri[t]); size.set(r, (size.get(r) || 0) + 1); }
  const kept = [];
  let dropped = 0, fragments = 0;
  for (const [, sz] of size) if (sz < minFaces) fragments++;
  for (let t = 0; t < tri.length; t += 3) {
    if (size.get(find(tri[t])) < minFaces) { dropped++; continue; }
    kept.push(tri[t], tri[t + 1], tri[t + 2]);
  }
  idxAcc.setArray(new Uint32Array(kept));
  if (dropped) log(`dropped ${fragments} fragments under ${minFaces} faces (${dropped} faces, ${(100 * dropped / (tri.length / 3)).toFixed(1)}%)`);
}

/* ---------------------------------------------------------------- orientation */
/**
 * Make every triangle wind the same way as its neighbours, facing outward.
 *
 * A splat-to-mesh isosurface arrives with 3.5% of its shared edges traversed in the
 * same direction by both faces, which is to say one of the two is inside out, and
 * simplification amplifies that to over 10%. Three things go wrong downstream, none of
 * them obviously: a front-face-only renderer culls the inside-out faces and the dark
 * background shows through as specks; vertex normals, being area-weighted sums of
 * face normals, get cancelled wherever neighbours disagree and the shading goes soft
 * and wrong; and the cube atlas, which picks a tile by face normal, sends an inside-out
 * face to the opposite tile.
 *
 * The repair is the classic one. Walk the faces across shared edges; a neighbour that
 * traverses the shared edge in the same direction is flipped. That makes each
 * connected component self-consistent. Whether a component is consistently inside out
 * is then decided by whether its faces mostly point away from its own centroid, and
 * the whole component is flipped if not.
 */
function orientConsistently(prim, log) {
  const pos = prim.getAttribute('POSITION');
  const idxAcc = prim.getIndices();
  const tri = Uint32Array.from(idxAcc.getArray());
  const faces = tri.length / 3;

  // edge -> faces that use it (unordered key)
  const key = (i, j) => (i < j ? i * 4294967296 + j : j * 4294967296 + i);
  const edgeFaces = new Map();
  for (let f = 0; f < faces; f++) {
    for (let k = 0; k < 3; k++) {
      const kk = key(tri[f * 3 + k], tri[f * 3 + (k + 1) % 3]);
      const list = edgeFaces.get(kk);
      if (list) list.push(f); else edgeFaces.set(kk, [f]);
    }
  }
  // does face f traverse edge (a->b) in that direction?
  const traverses = (f, a, b) => {
    for (let k = 0; k < 3; k++) if (tri[f * 3 + k] === a && tri[f * 3 + (k + 1) % 3] === b) return true;
    return false;
  };
  const flip = (f) => { const t = tri[f * 3 + 1]; tri[f * 3 + 1] = tri[f * 3 + 2]; tri[f * 3 + 2] = t; };

  const visited = new Uint8Array(faces);
  const comp = new Int32Array(faces).fill(-1);
  const wasFlipped = new Uint8Array(faces);
  let flipped = 0, components = 0;
  const stack = [];
  for (let seed = 0; seed < faces; seed++) {
    if (visited[seed]) continue;
    const c = components++;
    visited[seed] = 1; comp[seed] = c; stack.push(seed);
    while (stack.length) {
      const f = stack.pop();
      for (let k = 0; k < 3; k++) {
        const a = tri[f * 3 + k], b = tri[f * 3 + (k + 1) % 3];
        const across = edgeFaces.get(key(a, b));
        // Propagate across MANIFOLD edges only. The splat mesh has 8% of its edges
        // shared by three to seven faces, and walking through one of those hands the
        // patch beyond it an orientation that agrees with whichever face happened to
        // be visited first, not with its manifold neighbours. Measured: letting the
        // walk cross them left 11% of manifold edges inconsistent after "repair".
        // Stopping at them splits the mesh into patches that are each consistent
        // inside, and the outward test below orients every patch separately.
        if (across.length !== 2) continue;
        for (const g of across) {
          if (g === f || visited[g]) continue;
          // f goes a->b; a consistent neighbour goes b->a. Same direction means flip.
          if (traverses(g, a, b)) { flip(g); flipped++; wasFlipped[g] = 1; }
          visited[g] = 1; comp[g] = c; stack.push(g);
        }
      }
    }
  }

  // Which way should each patch face? Two sources of truth, used by size.
  //
  // The mesh arrived 96% consistently wound, so for any patch the ORIGINAL winding of
  // its majority is an excellent prior: if the walk ended up flipping more than half a
  // patch, it was the seed that was inside out, and the patch goes back. This is what
  // keeps the thousands of small patches right; a centroid test on twenty faces is a
  // coin toss, and trusting it left the surface full of culled holes.
  //
  // For a LARGE patch the centroid test is reliable and catches the one case the prior
  // cannot: a whole shell that was consistently inside out from the start.
  const patchSize = new Int32Array(components), patchFlipped = new Int32Array(components);
  for (let f = 0; f < faces; f++) { patchSize[comp[f]]++; if (wasFlipped[f]) patchFlipped[comp[f]]++; }
  let restored = 0;
  for (let f = 0; f < faces; f++) {
    const m = comp[f];
    if (patchFlipped[m] * 2 > patchSize[m]) { flip(f); restored++; }
  }

  // outward test, large patches only, area weighted
  const LARGE = 500;
  const cx = new Float64Array(components), cy = new Float64Array(components), cz = new Float64Array(components), cn = new Float64Array(components);
  const v = [0, 0, 0];
  for (let f = 0; f < faces; f++) {
    for (let k = 0; k < 3; k++) {
      pos.getElement(tri[f * 3 + k], v);
      cx[comp[f]] += v[0]; cy[comp[f]] += v[1]; cz[comp[f]] += v[2]; cn[comp[f]]++;
    }
  }
  const out = new Float64Array(components);
  const a = [0, 0, 0], b = [0, 0, 0], c3 = [0, 0, 0];
  for (let f = 0; f < faces; f++) {
    pos.getElement(tri[f * 3], a); pos.getElement(tri[f * 3 + 1], b); pos.getElement(tri[f * 3 + 2], c3);
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c3[0] - a[0], c3[1] - a[1], c3[2] - a[2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const m = comp[f];
    const d = [(a[0] + b[0] + c3[0]) / 3 - cx[m] / cn[m], (a[1] + b[1] + c3[1]) / 3 - cy[m] / cn[m], (a[2] + b[2] + c3[2]) / 3 - cz[m] / cn[m]];
    out[m] += n[0] * d[0] + n[1] * d[1] + n[2] * d[2];   // |n| carries the area weight
  }
  let inverted = 0;
  for (let f = 0; f < faces; f++) if (patchSize[comp[f]] >= LARGE && out[comp[f]] < 0) { flip(f); inverted++; }

  idxAcc.setArray(tri);
  log(`orientation: ${flipped.toLocaleString()} faces re-wound, ${restored.toLocaleString()} restored to the original majority, ${inverted.toLocaleString()} turned outward in large patches, ${components} patch${components === 1 ? '' : 'es'}`);
}

/* ------------------------------------------------------------------ smoothing */
/**
 * Taubin smoothing: a Laplacian step forward, then a slightly larger one back.
 *
 * A splat-to-mesh isosurface is textured with the lumps of the gaussians it passed
 * through, and under a plain material the product looks like it was upholstered in
 * gravel. Plain Laplacian smoothing would remove that but also shrink the mesh a little
 * per pass, which on a metric model is a dimensional error. Taubin's second step with a
 * negative factor a touch larger than the first cancels the shrinkage to first order
 * while keeping the low-pass effect: pits go, proportions stay.
 *
 * Runs before normals exist, so nothing has to be recomputed. Off by default, because
 * the Hunyuan3D path arrives smooth and would only lose crease definition.
 */
function taubin(prim, passes, log) {
  if (!passes) return;
  const posAcc = prim.getAttribute('POSITION');
  const tri = prim.getIndices().getArray();
  const n = posAcc.getCount();
  let pos = Float32Array.from(posAcc.getArray());

  const deg = new Uint32Array(n);
  for (let t = 0; t < tri.length; t += 3) { deg[tri[t]] += 2; deg[tri[t + 1]] += 2; deg[tri[t + 2]] += 2; }
  const start = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) start[i + 1] = start[i] + deg[i];
  const adj = new Uint32Array(start[n]);
  const cur = Uint32Array.from(start.subarray(0, n));
  for (let t = 0; t < tri.length; t += 3) {
    const [a, b, c] = [tri[t], tri[t + 1], tri[t + 2]];
    adj[cur[a]++] = b; adj[cur[a]++] = c; adj[cur[b]++] = a; adj[cur[b]++] = c; adj[cur[c]++] = a; adj[cur[c]++] = b;
  }

  const step = (src, factor) => {
    const out = Float32Array.from(src);
    for (let i = 0; i < n; i++) {
      const d = start[i + 1] - start[i];
      if (!d) continue;
      let mx = 0, my = 0, mz = 0;
      for (let k = start[i]; k < start[i + 1]; k++) {
        const j = adj[k];
        mx += src[j * 3]; my += src[j * 3 + 1]; mz += src[j * 3 + 2];
      }
      out[i * 3] += factor * (mx / d - src[i * 3]);
      out[i * 3 + 1] += factor * (my / d - src[i * 3 + 1]);
      out[i * 3 + 2] += factor * (mz / d - src[i * 3 + 2]);
    }
    return out;
  };
  for (let p = 0; p < passes; p++) {
    pos = step(pos, 0.5);
    pos = step(pos, -0.53);
  }
  posAcc.setArray(pos);
  log(`smoothed ${passes} Taubin passes`);
}

/* ------------------------------------------------------------ colour cleanup */
/**
 * Median-filter per-vertex colour across the mesh graph.
 *
 * A splat reconstruction meshes an isosurface through a gaussian field, and vertices
 * that land in interior or occluded pockets pick up the colour of interior gaussians,
 * which are dark. Measured on the reference chair: 18% of vertices below 0.15
 * luminance on a cream product, scattered as isolated specks through the cream. That
 * is textbook salt-and-pepper noise.
 *
 * Median, not mean, precisely because of the walnut legs. A mean would drag the dark
 * legs toward the cream shell and the cream toward the legs, smearing the one real
 * material boundary in the model. A median over a neighbourhood that is mostly cream
 * returns cream, over a neighbourhood that is mostly walnut returns walnut, and only
 * moves a value that is an outlier among its own neighbours.
 */
function despeckleColours(prim, log, passes = 2) {
  const colAcc = prim.getAttribute('COLOR_0');
  if (!colAcc) return;
  const tri = prim.getIndices().getArray();
  const n = colAcc.getCount();
  let col = Float32Array.from(colAcc.getArray());

  // vertex adjacency, built once
  const deg = new Uint32Array(n);
  for (let t = 0; t < tri.length; t += 3) {
    deg[tri[t]] += 2; deg[tri[t + 1]] += 2; deg[tri[t + 2]] += 2;
  }
  const start = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) start[i + 1] = start[i] + deg[i];
  const adj = new Uint32Array(start[n]);
  const cursor = Uint32Array.from(start.subarray(0, n));
  for (let t = 0; t < tri.length; t += 3) {
    const [a, b, c] = [tri[t], tri[t + 1], tri[t + 2]];
    adj[cursor[a]++] = b; adj[cursor[a]++] = c;
    adj[cursor[b]++] = a; adj[cursor[b]++] = c;
    adj[cursor[c]++] = a; adj[cursor[c]++] = b;
  }

  const lum = (arr, i) => 0.2126 * arr[i * 3] + 0.7152 * arr[i * 3 + 1] + 0.0722 * arr[i * 3 + 2];
  const darkBefore = (() => { let d = 0; for (let i = 0; i < n; i++) if (lum(col, i) < 0.15) d++; return d / n; })();

  const buf = [];
  for (let pass = 0; pass < passes; pass++) {
    const next = Float32Array.from(col);
    for (let i = 0; i < n; i++) {
      buf.length = 0;
      buf.push(i);
      for (let k = start[i]; k < start[i + 1]; k++) buf.push(adj[k]);
      if (buf.length < 4) continue;
      // rank the neighbourhood by luminance and take the middle one's whole colour,
      // rather than taking a per-channel median, which can invent a colour that no
      // vertex actually had
      buf.sort((x, y) => lum(col, x) - lum(col, y));
      const m = buf[buf.length >> 1];
      next[i * 3] = col[m * 3];
      next[i * 3 + 1] = col[m * 3 + 1];
      next[i * 3 + 2] = col[m * 3 + 2];
    }
    col = next;
  }

  const darkAfter = (() => { let d = 0; for (let i = 0; i < n; i++) if (lum(col, i) < 0.15) d++; return d / n; })();
  colAcc.setArray(col);
  log(`despeckled colour: ${(darkBefore * 100).toFixed(1)}% very dark vertices -> ${(darkAfter * 100).toFixed(1)}%`);
}

/**
 * Divide the studio lighting out of the photo colour.
 *
 * A reconstruction's vertex colour is what the camera saw: albedo times lighting. Ship
 * that and the viewer lights it again, so the bright shell reads as chalk and the
 * underside of the seat as a stain. The usual fix needs a model. This one needs only
 * what the mesh already has: normals.
 *
 * Directional lighting on a mostly-Lambertian product is, to first order, a function of
 * surface orientation alone. So fit log-luminance against the normal (an order-1
 * spherical harmonic: constant plus a dot with one direction), per vertex, with each
 * segmented region's median subtracted first so a walnut leg next to a cream shell
 * reads as two albedos rather than as one shadow. What the fit explains is lighting;
 * divide it out. What it cannot explain (ambient occlusion under the seat, the image's
 * own noise) is left alone, which is the honest failure mode.
 *
 * `strength` below 1 keeps a trace of the shading, because a fully flattened albedo
 * can look like plastic once the viewer relights it.
 */
function delight(prim, perTriangleRegion, log, strength = 0.85) {
  const colAcc = prim.getAttribute('COLOR_0');
  const nrmAcc = prim.getAttribute('NORMAL');
  if (!colAcc || !nrmAcc) return null;
  const col = Float32Array.from(colAcc.getArray());
  const nrm = nrmAcc.getArray();
  const tri = prim.getIndices().getArray();
  const n = colAcc.getCount();

  // vertex -> region, by majority over the triangles that use it
  const votes = new Map();
  for (let t = 0; t < tri.length; t += 3) {
    const r = perTriangleRegion[t / 3];
    for (let k = 0; k < 3; k++) {
      const v = tri[t + k];
      let m = votes.get(v);
      if (!m) votes.set(v, m = new Map());
      m.set(r, (m.get(r) || 0) + 1);
    }
  }
  const region = new Int32Array(n).fill(0);
  for (const [v, m] of votes) {
    let best = -1, bestN = -1;
    for (const [r, c] of m) if (c > bestN) { bestN = c; best = r; }
    region[v] = best;
  }

  const lum = (i) => 0.2126 * col[i * 3] + 0.7152 * col[i * 3 + 1] + 0.0722 * col[i * 3 + 2];
  const logL = new Float64Array(n);
  for (let i = 0; i < n; i++) logL[i] = Math.log(Math.max(lum(i), 1e-3));

  // per-region median log-luminance, so the fit sees shading rather than albedo
  const byRegion = new Map();
  for (let i = 0; i < n; i++) {
    let a = byRegion.get(region[i]);
    if (!a) byRegion.set(region[i], a = []);
    a.push(logL[i]);
  }
  const median = new Map();
  for (const [r, a] of byRegion) { a.sort((x, y) => x - y); median.set(r, a[a.length >> 1]); }

  // least squares: y = c0 + c1 nx + c2 ny + c3 nz, via the 4x4 normal equations
  const A = new Float64Array(16), b = new Float64Array(4);
  let ssTot = 0, meanY = 0;
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) { ys[i] = logL[i] - median.get(region[i]); meanY += ys[i]; }
  meanY /= n;
  for (let i = 0; i < n; i++) {
    const x = [1, nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]];
    for (let r = 0; r < 4; r++) {
      b[r] += x[r] * ys[i];
      for (let c = 0; c < 4; c++) A[r * 4 + c] += x[r] * x[c];
    }
    ssTot += (ys[i] - meanY) ** 2;
  }
  const coef = solve4(A, b);
  if (!coef) { log('de-light: degenerate fit, skipped'); return null; }

  // how much of the within-region brightness variation orientation explains
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const pred = coef[0] + coef[1] * nrm[i * 3] + coef[2] * nrm[i * 3 + 1] + coef[3] * nrm[i * 3 + 2];
    ssRes += (ys[i] - pred) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  // The bar was 4% while the splat mesh still had its ghost sheets, whose normals
  // made the fit meaningless and the stage a no-op. With a carved skin the normals
  // are real and a 7% fit went through, and it cost the walnut legs: a weak fit
  // still brightens every surface that happens to face away from its guessed key
  // light. Below 15% the shading is not explained by orientation; leave it.
  if (r2 < 0.15) {
    log(`de-light: orientation explains only ${(r2 * 100).toFixed(0)}% of the shading, left as photographed`);
    return { r2, applied: false };
  }

  // divide the fitted shading out, normalised so the mean brightness is unchanged
  let meanS = 0;
  const shade = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    shade[i] = Math.exp(coef[1] * nrm[i * 3] + coef[2] * nrm[i * 3 + 1] + coef[3] * nrm[i * 3 + 2]);
    meanS += shade[i];
  }
  meanS /= n;
  for (let i = 0; i < n; i++) {
    const factor = 1 / ((shade[i] / meanS) ** strength);
    for (let k = 0; k < 3; k++) col[i * 3 + k] = Math.min(1, Math.max(0, col[i * 3 + k] * factor));
  }
  colAcc.setArray(col);

  const dirLen = Math.hypot(coef[1], coef[2], coef[3]) || 1;
  const dir = [coef[1] / dirLen, coef[2] / dirLen, coef[3] / dirLen].map((v) => v.toFixed(2));
  log(`de-light: orientation explained ${(r2 * 100).toFixed(0)}% of the shading, key light from [${dir.join(', ')}], removed at ${strength}`);
  // the same correction as a function of a normal, for bakes that sample the dense
  // mesh per texel and never see these vertex colours
  const factorFor = (nx, ny, nz) => 1 / ((Math.exp(coef[1] * nx + coef[2] * ny + coef[3] * nz) / meanS) ** strength);
  return { r2, applied: true, light: dir, factorFor };
}

/** Gaussian elimination with partial pivoting, for the one 4x4 system above. */
function solve4(A, b) {
  const M = Array.from({ length: 4 }, (_, r) => [...A.slice(r * 4, r * 4 + 4), b[r]]);
  for (let c = 0; c < 4; c++) {
    let p = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 4; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 5; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, r) => row[4] / row[r]);
}

/* ------------------------------------------------------------- segmentation */
/**
 * Group triangles into regions a person would think of as separate parts.
 *
 * Connected-component labelling is not enough here. A reconstruction of a chair is
 * usually a single watertight shell, so every triangle is one component and "select
 * a part" would mean "select everything". What actually separates a seat from a back
 * from a leg on a smooth mesh is curvature: a crease.
 *
 * So this floods across shared edges and refuses to cross one where the dihedral
 * angle exceeds a threshold. Smooth surfaces grow into one region, hard edges stop
 * the flood, and the result is roughly the set of faces a person would click as "the
 * seat".
 */
function segment(prim, { creaseDegrees = 38, minShare = 0.004, maxRegions = 24, colourSplit = 0 }, log) {
  const pos = prim.getAttribute('POSITION');
  const tri = prim.getIndices().getArray();
  const triCount = tri.length / 3;

  // face normals and areas
  const fn = new Float32Array(triCount * 3);
  const area = new Float64Array(triCount);
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
  for (let t = 0; t < triCount; t++) {
    pos.getElement(tri[t * 3], a); pos.getElement(tri[t * 3 + 1], b); pos.getElement(tri[t * 3 + 2], c);
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const f = [e1[1] * e2[2] - e1[2] * e2[1],
               e1[2] * e2[0] - e1[0] * e2[2],
               e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(f[0], f[1], f[2]) || 1;
    area[t] = l / 2;
    fn[t * 3] = f[0] / l; fn[t * 3 + 1] = f[1] / l; fn[t * 3 + 2] = f[2] / l;
  }

  // Per-triangle colour, when the product's own colour is on the mesh.
  //
  // A crease is not what a person means by a part. The swoosh on a shoe, the walnut
  // legs under a boucle shell, the glow down a figure's flanks: all of them are
  // MATERIAL boundaries on a smooth surface, and a crease flood runs straight
  // through them. Where the colour changes sharply the flood should stop too.
  //
  // The comparison is in a small opponent space (lightness, red-green, blue-yellow)
  // on sRGB-encoded values, which is close enough to how a difference looks and much
  // cheaper than a real Lab conversion. Lightness is weighted at half: a shadowed
  // fold of one material must not read as a boundary, while a hue change must.
  const colAttr = prim.getAttribute('COLOR_0');
  let triCol = null, triMat = null;
  if (colAttr && colourSplit > 0) {
    const col = colAttr.getArray();
    triCol = new Float32Array(triCount * 3);
    const srgb = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
    for (let t = 0; t < triCount; t++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < 3; k++) {
        const v = tri[t * 3 + k];
        r += srgb(col[v * 3]); g += srgb(col[v * 3 + 1]); b += srgb(col[v * 3 + 2]);
      }
      r /= 3; g /= 3; b /= 3;
      triCol[t * 3] = 0.5 * (0.299 * r + 0.587 * g + 0.114 * b);
      triCol[t * 3 + 1] = r - g;
      triCol[t * 3 + 2] = b - (r + g) / 2;
    }

    // Materials are found GLOBALLY, not edge by edge. A flood that stops only where
    // neighbouring colours differ still walks from cream to walnut along a gradual
    // path, and on a photo-derived texture there is always a gradual path: measured
    // on the shoe, an edge test found five regions at one threshold and one at the
    // next. Clustering the whole surface first gives every triangle a material label
    // that no path can smear, and the flood then refuses to cross a label change.
    //
    // k-means, area-weighted, seeded far apart (k-means++ in spirit), with clusters
    // closer than the split threshold merged afterwards, so a product that really is
    // one material ends up with one label and the crease pass does its usual work.
    const K = 6;
    const centres = [];
    {
      let biggest = 0;
      for (let t = 1; t < triCount; t++) if (area[t] > area[biggest]) biggest = t;
      centres.push([triCol[biggest * 3], triCol[biggest * 3 + 1], triCol[biggest * 3 + 2]]);
      while (centres.length < K) {
        let far = -1, farD = -1;
        for (let t = 0; t < triCount; t += 7) {
          let d = Infinity;
          for (const c of centres) d = Math.min(d, (triCol[t * 3] - c[0]) ** 2 + (triCol[t * 3 + 1] - c[1]) ** 2 + (triCol[t * 3 + 2] - c[2]) ** 2);
          d *= area[t];
          if (d > farD) { farD = d; far = t; }
        }
        if (far < 0) break;
        centres.push([triCol[far * 3], triCol[far * 3 + 1], triCol[far * 3 + 2]]);
      }
    }
    triMat = new Uint8Array(triCount);
    for (let iter = 0; iter < 12; iter++) {
      const sum = centres.map(() => [0, 0, 0, 0]);
      for (let t = 0; t < triCount; t++) {
        let best = 0, bestD = Infinity;
        for (let c = 0; c < centres.length; c++) {
          const d = (triCol[t * 3] - centres[c][0]) ** 2 + (triCol[t * 3 + 1] - centres[c][1]) ** 2 + (triCol[t * 3 + 2] - centres[c][2]) ** 2;
          if (d < bestD) { bestD = d; best = c; }
        }
        triMat[t] = best;
        const a = area[t], acc = sum[best];
        acc[0] += triCol[t * 3] * a; acc[1] += triCol[t * 3 + 1] * a; acc[2] += triCol[t * 3 + 2] * a; acc[3] += a;
      }
      let moved = 0;
      for (let c = 0; c < centres.length; c++) {
        if (!sum[c][3]) continue;
        const nc = [sum[c][0] / sum[c][3], sum[c][1] / sum[c][3], sum[c][2] / sum[c][3]];
        moved += Math.hypot(nc[0] - centres[c][0], nc[1] - centres[c][1], nc[2] - centres[c][2]);
        centres[c] = nc;
      }
      if (moved < 1e-4) break;
    }
    // merge clusters that are not far enough apart to be different materials
    const map = centres.map((_, i) => i);
    for (let i = 0; i < centres.length; i++) {
      for (let j = 0; j < i; j++) {
        if (map[i] !== i || map[j] !== j) continue;
        const d = Math.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1], centres[i][2] - centres[j][2]);
        if (d < colourSplit) map[i] = j;
      }
    }
    const compact = new Map();
    for (let i = 0; i < map.length; i++) { let r = i; while (map[r] !== r) r = map[r]; if (!compact.has(r)) compact.set(r, compact.size); map[i] = compact.get(r); }
    for (let t = 0; t < triCount; t++) triMat[t] = map[triMat[t]];
    if (compact.size < 2) triMat = null;
  }


  // edge -> the (at most two) triangles that use it
  const edges = new Map();
  const key = (i, j) => (i < j ? i * 4294967296 + j : j * 4294967296 + i);
  for (let t = 0; t < triCount; t++) {
    const i0 = tri[t * 3], i1 = tri[t * 3 + 1], i2 = tri[t * 3 + 2];
    for (const [x, y] of [[i0, i1], [i1, i2], [i2, i0]]) {
      const k = key(x, y);
      const e = edges.get(k);
      if (e === undefined) edges.set(k, t);
      else if (typeof e === 'number') edges.set(k, [e, t]);
    }
  }
  const neighbours = Array.from({ length: triCount }, () => []);
  for (const e of edges.values()) {
    if (Array.isArray(e)) { neighbours[e[0]].push(e[1]); neighbours[e[1]].push(e[0]); }
  }

  const cosLimit = Math.cos((creaseDegrees * Math.PI) / 180);
  const label = new Int32Array(triCount).fill(-1);
  let next = 0;
  const stack = [];
  for (let seed = 0; seed < triCount; seed++) {
    if (label[seed] !== -1) continue;
    const id = next++;
    label[seed] = id;
    stack.push(seed);
    while (stack.length) {
      const t = stack.pop();
      for (const u of neighbours[t]) {
        if (label[u] !== -1) continue;
        const d = fn[t * 3] * fn[u * 3] + fn[t * 3 + 1] * fn[u * 3 + 1] + fn[t * 3 + 2] * fn[u * 3 + 2];
        if (d < cosLimit) continue;          // a crease: do not cross it
        if (triMat && triMat[t] !== triMat[u]) continue;      // a material boundary
        label[u] = id;
        stack.push(u);
      }
    }
  }

  // Measure what came out, then fold the specks away. A crease flood always leaves a
  // long tail of two-triangle slivers along the creases themselves, and offering a
  // person 4,000 clickable regions is the same as offering them none.
  const stats = Array.from({ length: next }, () => ({ tris: 0, area: 0, min: [1e9, 1e9, 1e9], max: [-1e9, -1e9, -1e9] }));
  void triCol;
  const v = [0, 0, 0];
  for (let t = 0; t < triCount; t++) {
    const s = stats[label[t]];
    s.tris++; s.area += area[t];
    for (let k = 0; k < 3; k++) {
      pos.getElement(tri[t * 3 + k], v);
      for (let d = 0; d < 3; d++) { s.min[d] = Math.min(s.min[d], v[d]); s.max[d] = Math.max(s.max[d], v[d]); }
    }
  }
  const total = stats.reduce((n, s) => n + s.area, 0) || 1;
  const keep = stats
    .map((s, id) => ({ id, ...s }))
    .filter((s) => s.area / total >= minShare)
    .sort((x, y) => y.area - x.area)
    .slice(0, maxRegions);
  log(`segmented into ${next} raw regions, keeping ${keep.length} above ${(minShare * 100).toFixed(1)}% of surface`);

  // every dropped region joins whichever kept region it touches most
  const kept = new Map(keep.map((s, i) => [s.id, i]));
  const remap = new Int32Array(next).fill(-1);
  for (const [oldId, i] of kept) remap[oldId] = i;
  for (let pass = 0; pass < 8; pass++) {
    let changed = 0;
    for (let t = 0; t < triCount; t++) {
      if (remap[label[t]] !== -1) continue;
      for (const u of neighbours[t]) {
        if (remap[label[u]] !== -1) { remap[label[t]] = remap[label[u]]; changed++; break; }
      }
    }
    if (!changed) break;
  }
  // anything still orphaned lands in the largest region rather than nowhere
  for (let i = 0; i < next; i++) if (remap[i] === -1) remap[i] = 0;

  const perTriangle = new Uint16Array(triCount);
  for (let t = 0; t < triCount; t++) perTriangle[t] = remap[label[t]];

  // Number the repeats. Six regions all called "Upper" is worse than useless in a
  // list you are meant to click through: you cannot tell which one you already did.
  // The mean colour of each kept region, for naming and for the studio's swatches.
  const kept0 = new Map(keep.map((s, i) => [s.id, i]));
  const meanCol = keep.map(() => [0, 0, 0, 0]);
  if (colAttr) {
    const col = colAttr.getArray();
    const srgb2 = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
    for (let t = 0; t < triCount; t++) {
      const i = kept0.get(label[t]);
      if (i === undefined) continue;
      const m = meanCol[i];
      for (let k = 0; k < 3; k++) {
        const v = tri[t * 3 + k];
        m[0] += srgb2(col[v * 3]) * area[t]; m[1] += srgb2(col[v * 3 + 1]) * area[t]; m[2] += srgb2(col[v * 3 + 2]) * area[t];
      }
      m[3] += 3 * area[t];
    }
  }
  const hexOf = (m) => (m[3] ? '#' + [0, 1, 2].map((i) => Math.round(255 * Math.min(1, Math.max(0, m[i] / m[3]))).toString(16).padStart(2, '0')).join('') : null);

  const seen = new Map();
  const regions = keep.map((s, i) => {
    const base = colourName(hexOf(meanCol[i])) || nameRegion(s, keep[0]);
    const n = (seen.get(base) || 0) + 1;
    seen.set(base, n);
    return {
      id: i,
      label: base,
      photoColour: hexOf(meanCol[i]),
      triangles: 0,
      area: 0,
      bbox: { min: s.min, max: s.max },
    };
  });
  for (const r of regions) {
    if (seen.get(r.label) > 1) {
      const n = (seen.get(`#${r.label}`) || 0) + 1;
      seen.set(`#${r.label}`, n);
      r.label = `${r.label} ${n}`;
    }
  }
  for (let t = 0; t < triCount; t++) { regions[perTriangle[t]].triangles++; regions[perTriangle[t]].area += area[t]; }
  for (const r of regions) r.share = r.area / total;

  return { perTriangle, regions, creaseDegrees };
}

/**
 * Name a part by the colour it actually is.
 *
 * "Walnut leg" and "Mint panel" beat "Base 3" for the same reason the geometric namer
 * exists: someone is clicking through a list and needs to know which one they already
 * did. Only strong, nameable colours get a name; anything muddy falls back to the
 * shape-based name, which is honest about knowing less.
 */
function colourName(hex) {
  if (!hex) return null;
  const r = parseInt(hex.slice(1, 3), 16) / 255, g = parseInt(hex.slice(3, 5), 16) / 255, b = parseInt(hex.slice(5, 7), 16) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const sat = mx > 0 ? (mx - mn) / mx : 0;
  const lum = 0.299 * r + 0.587 * g + 0.114 * b;
  // Absolute chroma as well as relative: at luminance 0.1 a two-unit difference is
  // 15% "saturation" and no colour at all, which is how a near-black figure came out
  // named Purple and Blue.
  if (mx - mn < 0.04) {
    if (lum > 0.82) return 'White';
    if (lum > 0.55) return 'Light grey';
    if (lum < 0.12) return 'Black';
    return null;                      // mid grey says nothing: let the shape name it
  }
  let h = 0;
  const d = mx - mn;
  if (mx === r) h = ((g - b) / d + 6) % 6;
  else if (mx === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  h *= 60;
  const warm = h >= 10 && h < 75;
  if (lum < 0.3 && warm && sat > 0.25) return 'Walnut';   // dark warm brown, worth its own name
  // A tint, not a colour: a cream boucle is 18% saturated and calling it Orange is
  // worse than calling it nothing. Between neutral and bold, name the tint.
  if (sat < 0.3) {
    if (warm) return lum > 0.45 ? 'Cream' : 'Taupe';
    if (h < 160) return lum > 0.5 ? 'Mint' : 'Sage';
    if (h < 200) return 'Pale teal';
    if (h < 290) return lum > 0.5 ? 'Pale blue' : 'Slate';
    return lum > 0.5 ? 'Blush' : 'Mauve';
  }
  if (h < 15 || h >= 345) return lum < 0.35 ? 'Deep red' : 'Red';
  if (h < 45) return lum < 0.4 ? 'Bronze' : 'Orange';
  if (h < 70) return 'Yellow';
  if (h < 100) return 'Lime';
  if (h < 160) return lum > 0.6 ? 'Mint' : 'Green';
  if (h < 200) return 'Teal';
  if (h < 250) return 'Blue';
  if (h < 290) return 'Purple';
  return 'Pink';
}

/** A guessable name beats "Region 7" when someone is clicking around. */
function nameRegion(s, biggest) {
  const h = s.max[1] - s.min[1], w = s.max[0] - s.min[0], d = s.max[2] - s.min[2];
  const top = s.min[1] > biggest.max[1] * 0.6;
  const low = s.max[1] < biggest.max[1] * 0.35;
  if (low && h > Math.max(w, d) * 1.4) return 'Leg';
  if (low) return 'Base';
  if (top && h > Math.max(w, d)) return 'Back';
  if (top) return 'Upper';
  if (h < Math.max(w, d) * 0.35) return 'Seat';
  return 'Body';
}

/* -------------------------------------------------------------- pose repair */
/**
 * Flat things come back lying down.
 *
 * The alignment's working assumption, "the model reliably puts the object upright"
 * (lib/align.mjs), holds for furniture and figures and fails for anything flat
 * photographed from above, which is how everyone photographs jewellery: the engine
 * rebuilds the pendant lying the way it lay on the table, the height scale then lands
 * on its thin axis, and the studio shows a walnut. Measured on a 6cm pendant: the
 * thin axis (its face) came back pointing [0.50, 0.87, -0.06], mostly up.
 *
 * Repair: when the reconstruction is decisively flat (thin/mid spread under FLAT),
 * rotate it so the thin axis faces the viewer and the long axis runs the way the
 * photograph's subject runs in frame. Two signs cannot come from flatness alone and
 * are measured instead:
 *
 *   up/down    - silhouette agreement with the photograph's own mask, over a small
 *                placement grid: a pendant's bail is at the top of exactly one of
 *                the two candidates;
 *   front/back - the conditioned face carries the photograph's detail and the
 *                invented back is smooth (measured: 1.33x the vertex-colour edge
 *                variation on the pendant), so the noisier side faces the viewer.
 *                Without vertex colour the toss is left as it lands; the projection
 *                paints whichever side fits.
 *
 * Runs before anything floor-relative, so the carve's depth views see the piece
 * face-on and the floor cut's "below" means below the product.
 */
const FLAT = 0.35;   // thin/mid principal spread; the pendant is 0.30, a chair ~0.8

function standUpright(prims, photoMask, maskSize, log) {
  // area-weighted PCA over triangle centroids, across every primitive
  const cent = []; const areas = [];
  let A = 0; const c = [0, 0, 0];
  for (const p of prims) {
    const pos = p.getAttribute('POSITION').getArray();
    const idx = p.getIndices().getArray();
    for (let t = 0; t < idx.length; t += 3) {
      const [i0, i1, i2] = [idx[t] * 3, idx[t + 1] * 3, idx[t + 2] * 3];
      const u = [pos[i1] - pos[i0], pos[i1 + 1] - pos[i0 + 1], pos[i1 + 2] - pos[i0 + 2]];
      const v = [pos[i2] - pos[i0], pos[i2 + 1] - pos[i0 + 1], pos[i2 + 2] - pos[i0 + 2]];
      const area = Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]) / 2;
      if (!area) continue;
      const q = [(pos[i0] + pos[i1] + pos[i2]) / 3, (pos[i0 + 1] + pos[i1 + 1] + pos[i2 + 1]) / 3, (pos[i0 + 2] + pos[i1 + 2] + pos[i2 + 2]) / 3];
      cent.push(q); areas.push(area); A += area;
      for (let k = 0; k < 3; k++) c[k] += area * q[k];
    }
  }
  if (!A) return;
  for (let k = 0; k < 3; k++) c[k] /= A;
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let t = 0; t < cent.length; t++) {
    const d = [cent[t][0] - c[0], cent[t][1] - c[1], cent[t][2] - c[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += areas[t] * d[i] * d[j];
  }
  const { evals, evecs } = eigen3(C);
  const order = [0, 1, 2].sort((i, j) => evals[j] - evals[i]);
  const spread = order.map((o) => Math.sqrt(Math.max(0, evals[o])));
  const ratio = spread[2] / (spread[1] || 1);
  if (ratio >= FLAT) return;   // not flat; the upright assumption stands

  const long = evecs[order[0]];
  const thin = evecs[order[2]];
  const tilt = Math.acos(Math.min(1, Math.abs(thin[2]))) * (180 / Math.PI);

  // Which way does the subject run in the photograph? 2D principal axis of the mask,
  // in y-up coordinates so it matches the silhouette camera. For a round brooch the
  // direction is arbitrary, and for a round brooch any roll is equally right.
  let mx = 0, my = 0, m = 0;
  for (let row = 0; row < maskSize; row++) {
    for (let col = 0; col < maskSize; col++) {
      if (!photoMask[row * maskSize + col]) continue;
      mx += col; my += (maskSize - 1 - row); m++;
    }
  }
  if (!m) return;
  mx /= m; my /= m;
  let sxx = 0, syy = 0, sxy = 0;
  for (let row = 0; row < maskSize; row++) {
    for (let col = 0; col < maskSize; col++) {
      if (!photoMask[row * maskSize + col]) continue;
      const dx = col - mx, dy = (maskSize - 1 - row) - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
  }
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  let dir = [Math.cos(theta), Math.sin(theta)];
  if (dir[1] < 0 || (dir[1] === 0 && dir[0] < 0)) dir = [-dir[0], -dir[1]];

  // Object basis: long axis, thin axis, and their completion, right-handed.
  const nrm = (v) => { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); };
  const a3 = nrm(thin);
  const dot = long[0] * a3[0] + long[1] * a3[1] + long[2] * a3[2];
  const a1 = nrm([long[0] - dot * a3[0], long[1] - dot * a3[1], long[2] - dot * a3[2]]);
  const a2 = [a3[1] * a1[2] - a3[2] * a1[1], a3[2] * a1[0] - a3[0] * a1[2], a3[0] * a1[1] - a3[1] * a1[0]];

  // Two candidates: the long axis runs with the mask's direction, or against it.
  // R maps object axes onto view axes (thin toward the camera), rows = target basis
  // dotted with object basis.
  const rotFor = (s) => {
    const b1 = [s * dir[0], s * dir[1], 0];
    const b3 = [0, 0, 1];
    const b2 = [b3[1] * b1[2] - b3[2] * b1[1], b3[2] * b1[0] - b3[0] * b1[2], b3[0] * b1[1] - b3[1] * b1[0]];
    return [0, 1, 2].map((r) => [0, 1, 2].map((k) =>
      [b1, b2, b3][0][r] * a1[k] + [b1, b2, b3][1][r] * a2[k] + [b1, b2, b3][2][r] * a3[k]));
  };

  // Score each by silhouette agreement over a small placement grid, on a triangle
  // subsample: a binary decision with a real margin does not need the full mesh.
  const first = prims[0];
  const pos0 = first.getAttribute('POSITION').getArray();
  const idx0 = first.getIndices().getArray();
  const stride = Math.max(1, Math.floor(idx0.length / 3 / 150_000)) * 3;
  const sub = new Uint32Array(3 * Math.ceil(idx0.length / stride));
  let ns = 0;
  for (let t = 0; t + 2 < idx0.length; t += stride) { sub[ns++] = idx0[t]; sub[ns++] = idx0[t + 1]; sub[ns++] = idx0[t + 2]; }
  const subIdx = sub.subarray(0, ns);
  const front = viewFor(0, 0);
  const scoreOf = (R) => {
    const rot = new Float32Array(pos0.length);
    for (let i = 0; i < pos0.length; i += 3) {
      for (let r = 0; r < 3; r++) rot[i + r] = R[r][0] * pos0[i] + R[r][1] * pos0[i + 1] + R[r][2] * pos0[i + 2];
    }
    let best = 0;
    for (const scale of [0.85, 1, 1.15]) {
      for (const dx of [-0.06, 0, 0.06]) {
        for (const dy of [-0.06, 0, 0.06]) {
          const s = iou(silhouette(rot, subIdx, { ...front, _proj: null }, maskSize, { scale, dx, dy }), photoMask);
          if (s > best) best = s;
        }
      }
    }
    return best;
  };
  const up = scoreOf(rotFor(1)), down = scoreOf(rotFor(-1));
  let R = rotFor(up >= down ? 1 : -1);

  // apply, to every primitive and to normals when they exist
  for (const p of prims) {
    for (const name of ['POSITION', 'NORMAL']) {
      const acc = p.getAttribute(name);
      if (!acc) continue;
      const arr = acc.getArray();
      for (let i = 0; i < arr.length; i += 3) {
        const [x, y, z] = [arr[i], arr[i + 1], arr[i + 2]];
        for (let r = 0; r < 3; r++) arr[i + r] = R[r][0] * x + R[r][1] * y + R[r][2] * z;
      }
      acc.setArray(arr);
    }
  }

  // front/back: the side with more vertex-colour variation is the photographed one
  let turned = '';
  const colAcc = first.getAttribute('COLOR_0');
  if (colAcc) {
    const nV = colAcc.getCount();
    const cols = new Float32Array(nV * 3);
    const el = [0, 0, 0, 0];
    for (let i = 0; i < nV; i++) { colAcc.getElement(i, el); cols[i * 3] = el[0]; cols[i * 3 + 1] = el[1]; cols[i * 3 + 2] = el[2]; }
    const posR = first.getAttribute('POSITION').getArray();
    let fw = 0, fd = 0, bw = 0, bd = 0;
    for (let t = 0; t < idx0.length; t += 3) {
      const [i0, i1, i2] = [idx0[t], idx0[t + 1], idx0[t + 2]];
      const p0 = posR.subarray(i0 * 3, i0 * 3 + 3), p1 = posR.subarray(i1 * 3, i1 * 3 + 3), p2 = posR.subarray(i2 * 3, i2 * 3 + 3);
      const u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
      const v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
      const cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const area = Math.hypot(...cr) / 2;
      if (!area) continue;
      const nz = cr[2] / (2 * area);
      if (Math.abs(nz) < 0.3) continue;
      let d = 0;
      for (const [a, b] of [[i0, i1], [i1, i2], [i2, i0]]) {
        for (let k = 0; k < 3; k++) d += Math.abs(cols[a * 3 + k] - cols[b * 3 + k]);
      }
      if (nz > 0) { fw += area; fd += area * d; } else { bw += area; bd += area * d; }
    }
    if (fw && bw && bd / bw > 1.15 * (fd / fw)) {
      // turn it around: 180 degrees about the vertical
      for (const p of prims) {
        for (const name of ['POSITION', 'NORMAL']) {
          const acc = p.getAttribute(name);
          if (!acc) continue;
          const arr = acc.getArray();
          for (let i = 0; i < arr.length; i += 3) { arr[i] = -arr[i]; arr[i + 2] = -arr[i + 2]; }
          acc.setArray(arr);
        }
      }
      turned = ', turned to face its detailed side forward';
    }
  }

  log(`flat reconstruction (thin/mid spread ${ratio.toFixed(2)}) was lying ${tilt.toFixed(0)} degrees from face-on: `
    + `stood upright (silhouette ${(Math.max(up, down) * 100).toFixed(0)}% vs ${(Math.min(up, down) * 100).toFixed(0)}% the other way up${turned})`);
}

/** Eigen decomposition of a symmetric 3x3, by Jacobi rotations. */
function eigen3(M) {
  const a = M.map((r) => r.slice());
  let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const mul = (X, Y) => X.map((_, i) => [0, 1, 2].map((j) => X[i][0] * Y[0][j] + X[i][1] * Y[1][j] + X[i][2] * Y[2][j]));
  for (let iter = 0; iter < 50; iter++) {
    let p = 0, q = 1, off = Math.abs(a[0][1]);
    if (Math.abs(a[0][2]) > off) { off = Math.abs(a[0][2]); p = 0; q = 2; }
    if (Math.abs(a[1][2]) > off) { off = Math.abs(a[1][2]); p = 1; q = 2; }
    if (off < 1e-12) break;
    const theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    R[p][p] = Math.cos(theta); R[q][q] = Math.cos(theta);
    R[p][q] = Math.sin(theta); R[q][p] = -Math.sin(theta);
    const Rt = [0, 1, 2].map((i) => R.map((r) => r[i]));
    const an = mul(mul(Rt, a), R);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) a[i][j] = an[i][j];
    V = mul(V, R);
  }
  return { evals: [a[0][0], a[1][1], a[2][2]], evecs: [0, 1, 2].map((j) => V.map((r) => r[j])) };
}

/* ------------------------------------------------------------------- public */
/**
 * Build a Blank from a raw reconstruction.
 *
 * targetHeight is in metres and is the one number that makes AR meaningful: a chair
 * placed at the wrong scale is a toy. tile is how much of the real world one texture
 * repeat covers, so 0.16 means a weave repeats every 16cm no matter how big the piece.
 */
export async function makeBlank(rawPath, outDir, {
  targetHeight = 0.78, targetWidth = null, targetDepth = null,
  tile = 0.16, triBudget = 60_000,
  creaseDegrees = null, minRegions = 6, smoothPasses = 0, carve = false, dense = false,
  colourSplit = 0.1, photoMask = null, photoMaskSize = 192, onProgress,
} = {}) {
  const log = (m) => onProgress?.({ stage: 'finish', detail: m });

  const io = new NodeIO();
  const doc = await io.read(rawPath);
  const root = doc.getRoot();

  const countTris = () => root.listMeshes().flatMap((m) => m.listPrimitives())
    .reduce((n, p) => n + (p.getIndices()
      ? p.getIndices().getCount() / 3
      : p.getAttribute('POSITION').getCount() / 3), 0);

  const rawTris = countTris();
  log(`raw mesh: ${rawTris.toLocaleString()} triangles`);

  // Cut the floor BEFORE simplifying, not after.
  //
  // The other way round spends the whole triangle budget on geometry that is about to
  // be deleted. Measured on a two-view reconstruction: simplify-then-cut left 3,197
  // triangles of a 60,000 budget, because the cut removed 147,985 of the 151,182 that
  // simplification had produced. Cutting first means the budget is spent on the
  // product.
  await MeshoptSimplifier.ready;
  await doc.transform(weld());
  for (const prim of root.listMeshes().flatMap((m) => m.listPrimitives())) dropGroundPlane(prim, log);

  // Flat reconstructions come back lying down; stand them up before anything
  // orientation-dependent runs, so the carve sees the piece face-on and the height
  // scale lands on its long axis. No-op for anything that is not decisively flat.
  if (photoMask) standUpright(root.listMeshes().flatMap((m) => m.listPrimitives()), photoMask, photoMaskSize, log);

  // The splat engine's mesh is a double wall with interior sheets (see lib/carve.mjs).
  // Rebuild its outer skin before spending the triangle budget on it.
  // Keep the raw mesh before the carve replaces it: it is the source of the detail
  // normal map baked alongside the colour atlas (lib/detail.mjs). The raw mesh, not
  // the carved hull: the hull at 4 mm voxels carries a staircase that three smoothing
  // passes do not remove and that baked as 20 degrees of speckle, while the raw mesh
  // has the figure's actual scales. Its own defects are rejected by the lookup: ghost
  // sheets face the wrong way and tunnel walls are edge-on, and neither is a
  // candidate for a coarse surface facing outward.
  let hiRes = null;
  let thinShare = null;
  // Detached pieces above the object. A caption strip or a patch of backdrop in one
  // of the photographs can come back from a multi-view reconstruction as a slab
  // floating over the head; the floor cut only looks below. Anything disconnected,
  // small next to the main body, and not touching its box is dropped.
  for (const prim of root.listMeshes().flatMap((m) => m.listPrimitives())) dropDetached(prim, log);
  if (dense && !carve) {
    // a clean mesh (Hunyuan3D shape with transferred colour): no carve, but the raw
    // mesh is still the source for the per-texel colour and relief bakes
    const first = root.listMeshes().flatMap((m) => m.listPrimitives())[0];
    hiRes = {
      positions: Float32Array.from(first.getAttribute('POSITION').getArray()),
      indices: Uint32Array.from(first.getIndices().getArray()),
      colours: first.getAttribute('COLOR_0') ? Float32Array.from(first.getAttribute('COLOR_0').getArray()) : null,
    };
  }
  if (carve) {
    const first = root.listMeshes().flatMap((m) => m.listPrimitives())[0];
    hiRes = {
      positions: Float32Array.from(first.getAttribute('POSITION').getArray()),
      indices: Uint32Array.from(first.getIndices().getArray()),
      colours: first.getAttribute('COLOR_0') ? Float32Array.from(first.getAttribute('COLOR_0').getArray()) : null,
    };
    for (const prim of root.listMeshes().flatMap((m) => m.listPrimitives())) {
      const carved = carveSkin(prim, log, typeof carve === 'object' ? carve : {});
      thinShare = carved.thinShare;
      // smooth the dense hull before simplification, where the voxel staircase is
      // still a small fraction of an edge; the coarse mesh inherits the result
      taubin(prim, smoothPasses, log);
    }
  }

  // Simplify to the budget, loosening the error tolerance until it gets there.
  //
  // meshopt treats `error` as a ceiling it will not cross, so a tight tolerance simply
  // stops early on a very dense input: the TripoSplat path arrives at six million
  // triangles and 0.002 left 324,900 of a 60,000 budget, five times over. Rather than
  // pick one number that suits neither input, ask for the budget and widen the licence
  // until the budget is met. Hunyuan3D meshes still land on the first pass at 0.002,
  // so nothing about the existing path changes.
  const afterCut = countTris();
  let simplified = afterCut;
  for (const error of [0.002, 0.01, 0.05, 0.2]) {
    await doc.transform(
      simplify({ simplifier: MeshoptSimplifier, ratio: Math.min(1, triBudget / simplified), error }),
    );
    simplified = countTris();
    if (simplified <= triBudget * 1.1) break;
  }
  log(`simplified ${afterCut.toLocaleString()} to ${simplified.toLocaleString()} triangles`);

  for (const p of root.listMeshes().flatMap((m) => m.listPrimitives())) {
    dedupeFaces(p, log);
    dropFragments(p, log);
    orientConsistently(p, log);
    if (!hiRes) taubin(p, smoothPasses, log);   // a carved hull was smoothed before simplification
  }

  const prims = root.listMeshes().flatMap((m) => m.listPrimitives());

  // Measure only vertices the triangles still reference. The accessor's own min/max
  // covers everything in the buffer, including the floor slab just cut away, and
  // scaling to that would size the chair by its shadow.
  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const p of prims) {
    const acc = p.getAttribute('POSITION');
    const list = p.getIndices()?.getArray();
    const count = list ? list.length : acc.getCount();
    const el = [0, 0, 0];
    for (let i = 0; i < count; i++) {
      acc.getElement(list ? list[i] : i, el);
      for (let k = 0; k < 3; k++) {
        bounds.min[k] = Math.min(bounds.min[k], el[k]);
        bounds.max[k] = Math.max(bounds.max[k], el[k]);
      }
    }
  }
  const size = bounds.max.map((v, i) => v - bounds.min[i]);

  // Per-axis scale when the real dimensions are known. Height alone leaves width and
  // depth wherever the reconstruction guessed, and a reconstruction's proportions are
  // approximate: a retailer's feed wants all three numbers, and a shopper asking
  // "will it fit in the alcove" is asking about the real bounding box, not about the
  // model's guess at it. Distorting the mesh to the true box trades a little shape
  // fidelity for dimensional honesty, which for AR is the right trade. Axes without a
  // given dimension follow the height scale, exactly as before.
  const sy = targetHeight / size[1];
  const sx = targetWidth > 0 ? targetWidth / size[0] : sy;
  const sz = targetDepth > 0 ? targetDepth / size[2] : sy;
  const centre = [(bounds.min[0] + bounds.max[0]) / 2, bounds.min[1], (bounds.min[2] + bounds.max[2]) / 2];
  if (sx !== sy || sz !== sy) {
    log(`scaled per axis to ${(size[0] * sx).toFixed(2)} x ${(size[2] * sz).toFixed(2)} x ${targetHeight} m (w x d x h)`);
    const skew = Math.max(sx, sy, sz) / Math.min(sx, sy, sz);
    if (skew > 1.25) log(`note: the given dimensions stretch the reconstruction ${((skew - 1) * 100).toFixed(0)}% anisotropically; if that looks wrong, check the numbers`);
  } else {
    log(`scaled to ${targetHeight} m tall (factor ${sy.toFixed(3)})`);
  }

  for (const p of prims) {
    // Bake the transform into the vertices. Keeping the node graph trivial matters
    // because USD and glTF disagree about how node transforms compose often enough
    // to be worth sidestepping entirely.
    const acc = p.getAttribute('POSITION');
    const arr = acc.getArray();
    for (let i = 0; i < arr.length; i += 3) {
      arr[i] = (arr[i] - centre[0]) * sx;
      arr[i + 1] = (arr[i + 1] - centre[1]) * sy;
      arr[i + 2] = (arr[i + 2] - centre[2]) * sz;
    }
    acc.setArray(arr);
    if (hiRes && p === prims[0]) {
      const h = hiRes.positions;
      for (let i = 0; i < h.length; i += 3) {
        h[i] = (h[i] - centre[0]) * sx;
        h[i + 1] = (h[i + 1] - centre[1]) * sy;
        h[i + 2] = (h[i + 2] - centre[2]) * sz;
      }
    }

    // an anisotropic scale bends existing normals (they transform by the inverse
    // transpose, not by the scale), so recompute rather than carry stale ones
    if (sx !== sy || sz !== sy) p.setAttribute('NORMAL', null);
    if (!p.getAttribute('NORMAL')) computeNormals(doc, p);

    // box-project UVs at a fixed texel density, so a weave reads at one scale
    // wherever it lands on the form
    const pos = p.getAttribute('POSITION');
    const nrm = p.getAttribute('NORMAL');
    const count = pos.getCount();
    const uv = new Float32Array(count * 2);
    const pv = [0, 0, 0], nv = [0, 0, 0];
    for (let i = 0; i < count; i++) {
      pos.getElement(i, pv); nrm.getElement(i, nv);
      const ax = Math.abs(nv[0]), ay = Math.abs(nv[1]), az = Math.abs(nv[2]);
      const axis = ax > ay && ax > az ? 0 : ay > az ? 1 : 2;
      uv[i * 2] = (axis === 0 ? pv[2] : pv[0]) / tile;
      uv[i * 2 + 1] = (axis === 1 ? pv[2] : pv[1]) / tile;
    }
    p.setAttribute('TEXCOORD_0', doc.createAccessor().setType('VEC2').setArray(uv));
  }

  // Colour cleanup before segmentation, so what the studio shows and what the baker
  // rasterises are the same values.
  for (const p of prims) despeckleColours(p, log);

  // Segmentation runs last, on the geometry that will actually ship, so a triangle
  // index in regions.json means the same triangle the browser raycasts against.
  //
  // The crease angle cannot be a constant. It was tuned at 38 degrees against plain
  // marching cubes output, where the stair-stepping supplies creases everywhere and
  // the flood stops constantly. Switching to surface net removed those artifacts,
  // which is the point of switching, and the same 38 degrees then found ONE region
  // covering the whole chair: nothing to select, nothing to paint separately.
  //
  // So walk the angle down until the split is actually useful. A smooth
  // reconstruction needs a far lower threshold than a faceted one, and the right
  // value depends on the mesh rather than on the setting that produced it.
  const main = prims[0];
  const ladder = creaseDegrees != null ? [creaseDegrees] : [38, 26, 18, 12, 8];
  let seg = null;
  for (const angle of ladder) {
    seg = segment(main, { creaseDegrees: angle, colourSplit }, log);
    const biggest = Math.max(...seg.regions.map((r) => r.share));
    // "useful" means enough parts AND no single part swallowing the whole object
    if (seg.regions.length >= minRegions && biggest < 0.7) {
      log(`crease angle ${angle} deg gives ${seg.regions.length} parts, largest ${(biggest * 100).toFixed(0)}%`);
      break;
    }
    if (angle !== ladder[ladder.length - 1]) {
      log(`crease angle ${angle} deg gave ${seg.regions.length} parts (largest ${(biggest * 100).toFixed(0)}%), going finer`);
    }
  }

  // Lighting comes out before the colour is baked, and after segmentation so the fit
  // can tell a dark part from a shadowed one.
  const main0 = prims[0];
  let delit = null;
  if (main0.getAttribute('COLOR_0')) {
    // No per-vertex "dark fleck" repair here, and that is deliberate. One was built
    // while the splat mesh still had its ghost sheets, and it worked on them; once
    // the carve removed the sheets the same rule found the walnut legs instead, as
    // 116 small dark clusters, and painted them cream. The carve fixed the cause.
    delit = delight(main0, seg.perTriangle, log);
  }

  // If the reconstruction carried colour, turn it into a real texture now.
  //
  // This has to happen here rather than at bake time because it is a property of the
  // Blank, not of the Look: one photograph produces one photo-colour map, and every
  // Look that uses it shares it. The atlas UVs are stored per TRIANGLE rather than per
  // vertex, because the cube atlas is unwelded (a vertex on a tile seam needs a
  // different UV in each tile) and so cannot be an attribute on the welded blank.
  let hasColour = false;
  let hasDetail = false, detailStrength = 0;
  let hasGlow = false, glowShare = 0;
  let denseBake = null;
  if (main0.getAttribute('COLOR_0')) {
    const at = cubeAtlas(
      main0.getAttribute('POSITION').getArray(),
      main0.getAttribute('NORMAL').getArray(),
      main0.getAttribute('COLOR_0').getArray(),
      main0.getIndices().getArray(),
    );
    // Colour comes from the DENSE mesh when there is one. Interpolating the coarse
    // mesh's thirty thousand vertex colours across the atlas throws away the seven
    // hundred thousand the reconstruction returned; sampling the raw mesh at each
    // texel's closest surface point keeps them, on the same atlas, for the cost of
    // a lookup the relief bake already does. The de-light correction is applied per
    // texel from the coarse normal, so the texture and the vertex colours agree.
    let baked;
    if (hiRes?.colours && !process.env.DENSE_SKIP) {
      // 2048 for the colour: the front photograph is 700 px across and a 1024 atlas
      // gave it about 300 texels; relief and glow stay at 1024, which is plenty for
      // normals and costs a quarter of the bytes
      const dense = bakeDetail(at, hiRes, 2048, { shade: delit?.factorFor || null });
      baked = { rgb: dense.colour, size: dense.size, coverage: dense.coverage };
      denseBake = dense;
      log(`photo colour baked from ${(hiRes.positions.length / 3).toLocaleString()} raw vertices, ${(dense.coverage * 100).toFixed(0)}% of the atlas covered`);
    } else {
      baked = bakeColours(at, 1024);
      log(`photo colour baked, ${(baked.coverage * 100).toFixed(0)}% of the atlas covered`);
    }
    if (process.env.BAKE_DEBUG) console.error('photo bake buffer', baked.rgb?.constructor?.name, baked.rgb?.length, 'size', baked.size, 'expected', baked.size * baked.size * 3);
    await sharp(baked.rgb, { raw: { width: baked.size, height: baked.size, channels: 3 } })
      .jpeg({ quality: 90, chromaSubsampling: '4:4:4' })
      .toFile(`${outDir}/photo.jpg`);
    await writeFile(`${outDir}/atlas.bin`, Buffer.from(at.uvs.buffer, at.uvs.byteOffset, at.uvs.byteLength));
    hasColour = true;

    // Glow. A product that emits light photographs as saturated colour far brighter
    // than everything around it, and shipping that as plain albedo is what makes a
    // lit figure or an LED read as painted plastic in AR. Measured on the Shin
    // Godzilla figure: body median luminance 0.017, the red glow 3 to 12x that at
    // saturation above 0.5, 2 to 4% of the vertices. On the cream chair: nothing
    // qualifies. Ramp from 3x to 6x so the edge of a glow fades rather than cuts,
    // and give up when more than a fifth of the surface qualifies, because then the
    // bright saturated colour IS the product (a red sofa) and not a light.
    {
      const col = main0.getAttribute('COLOR_0').getArray();
      const n = col.length / 3;
      const lum = (i) => 0.2126 * col[i * 3] + 0.7152 * col[i * 3 + 1] + 0.0722 * col[i * 3 + 2];
      const med = Float64Array.from({ length: n }, (_, i) => lum(i)).sort()[n >> 1] || 1e-3;
      const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
      // The ceiling a lit surface can reach is set by the whitest thing on the object:
      // nothing diffuse can be brighter than white under the same light. Saturated
      // colour above that ceiling is emitting. Against the median alone a pastel shoe
      // "glowed" yellow over 12% of itself; against its white midsole, nothing does.
      const whites = [];
      for (let i = 0; i < n; i++) {
        const r = col[i * 3], g = col[i * 3 + 1], b = col[i * 3 + 2];
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        if (mx > 0 && (mx - mn) / mx < 0.25) whites.push(lum(i));
      }
      let ceiling;
      if (whites.length > n * 0.01) { const w = Float64Array.from(whites).sort(); ceiling = w[Math.floor(w.length * 0.98)]; }
      else ceiling = med * 3;   // an object with nothing white on it: fall back to the median rule
      const glow = new Float32Array(n * 3);
      let lit = 0;
      for (let i = 0; i < n; i++) {
        const r = col[i * 3], g = col[i * 3 + 1], b = col[i * 3 + 2];
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const sat = mx > 0 ? (mx - mn) / mx : 0;
        const w = smooth(1.2, 2.0, lum(i) / ceiling) * smooth(3, 6, lum(i) / med) * smooth(0.35, 0.5, sat);
        if (w > 0.5) lit++;
        glow[i * 3] = r * w; glow[i * 3 + 1] = g * w; glow[i * 3 + 2] = b * w;
      }
      const share = lit / n;
      if (lit && share < 0.2) {
        const gb = bakeColours({ uvs: at.uvs, colours: (() => {
          // the atlas is unwelded; expand per-vertex glow to per-corner
          const tri = main0.getIndices().getArray();
          const out = new Float32Array(tri.length * 3);
          for (let i = 0; i < tri.length; i++) for (let k = 0; k < 3; k++) out[i * 3 + k] = glow[tri[i] * 3 + k];
          return out;
        })(), indices: at.indices }, 1024);
        await sharp(gb.rgb, { raw: { width: gb.size, height: gb.size, channels: 3 } })
          .jpeg({ quality: 88, chromaSubsampling: '4:4:4' })
          .toFile(`${outDir}/glow.jpg`);
        hasGlow = true;
        glowShare = share;
        log(`glow baked: ${(share * 100).toFixed(1)}% of the surface emits`);
      } else if (lit) {
        log(`glow skipped: ${(share * 100).toFixed(0)}% of the surface is bright and saturated, so that is the product's colour`);
      }
    }

    if (hiRes) {
      const det = denseBake || bakeDetail(at, hiRes, 1024);
      await sharp(det.rgb, { raw: { width: det.size, height: det.size, channels: 3 } })
        .resize(1024, 1024, { kernel: 'lanczos3' })
        .png({ compressionLevel: 9 })
        .toFile(`${outDir}/detail.png`);
      hasDetail = true;
      detailStrength = det.strength;
      log(`detail baked from ${(hiRes.indices.length / 3).toLocaleString()} raw triangles: mean ${det.strength.toFixed(1)} degrees of relief`);
    }
  }

  // Each part's own photographed colour: the median of its vertex colours, as a hex
  // the studio can hand to a repaint as the starting tint. A repaint then begins as
  // "this part, as it is" rather than as the catalogue's grey.
  if (main0.getAttribute('COLOR_0')) {
    const col = main0.getAttribute('COLOR_0').getArray();
    const tri = main0.getIndices().getArray();
    const perRegion = seg.regions.map(() => [[], [], []]);
    for (let t = 0; t < seg.perTriangle.length; t++) {
      const r = seg.perTriangle[t];
      if (!perRegion[r]) continue;
      for (let k = 0; k < 3; k++) { const v = tri[t * 3 + k]; for (let c = 0; c < 3; c++) perRegion[r][c].push(col[v * 3 + c]); }
    }
    const toSrgb = (c) => Math.round(255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055));
    for (const [i, lists] of perRegion.entries()) {
      if (!lists[0].length || seg.regions[i].photoColour) continue;   // segment named it already
      const med = lists.map((l) => { const a = Float32Array.from(l).sort(); return a[a.length >> 1]; });
      seg.regions[i].photoColour = '#' + med.map((c) => toSrgb(Math.min(1, Math.max(0, c))).toString(16).padStart(2, '0')).join('');
    }
  }

  await io.write(`${outDir}/blank.glb`, doc);
  await writeFile(`${outDir}/regions.json`, JSON.stringify({
    triangleCount: seg.perTriangle.length,
    // a Uint16 per triangle, base64'd: 60k triangles is 120 KB raw and 160 KB as
    // base64, against about 400 KB as a JSON array of numbers
    perTriangle: Buffer.from(seg.perTriangle.buffer).toString('base64'),
    regions: seg.regions,
  }));

  const meta = {
    createdAt: new Date().toISOString(),
    rawTriangles: rawTris,
    triangles: countTris(),
    targetHeight, tile, creaseDegrees: seg.creaseDegrees,
    dimensions: { width: size[0] * sx, height: size[1] * sy, depth: size[2] * sz },
    // raw -> blank: p_blank = (p_raw - centre) * scale, so a later pass can map the
    // raw voxel grid into the frame the photographs were aligned in
    frame: { centre, scale: [sx, sy, sz] },
    regions: seg.regions.length,
    hasColour,
    hasDetail,
    detailStrength,
    hasGlow,
    glowShare,
    thinShare,
    delight: delit,
  };
  await writeFile(`${outDir}/blank.json`, JSON.stringify(meta, null, 2));
  log(`blank ready: ${meta.triangles.toLocaleString()} triangles, ${meta.regions} regions`);
  return meta;
}
export { orientConsistently, taubin, segment };
