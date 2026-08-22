/**
 * Rebuild a mesh's outer skin from what can be seen of it.
 *
 * Why this exists. TripoSplat's mesh is a thin double wall: meshing a layer of flat
 * gaussians yields both sides of the layer, an outer sheet the camera saw and a ghost
 * sheet 10 to 15 mm behind it facing the other way, pinched together at hundreds of
 * points. Smoothing drags each pinch into a tunnel, and the tunnel walls render as
 * flecks. Repairing that face by face was tried at length (duplicate removal,
 * fragment removal, short-edge collapse, visibility culling with hole filling, three
 * kinds of vertex lifting) and each pass left the next symptom behind, because the
 * ghost sheet is woven into the outer one and cannot be unpicked locally.
 *
 * So instead of repairing the mesh, take what a camera would see of it and build a
 * new one. Render depth from directions spread over a sphere; any voxel that sits in
 * front of the nearest surface in any view is outside, and what survives is the
 * depth-carved visual hull: one closed outer surface, by construction free of
 * interior sheets, pinches and tunnels. Surface nets turn the voxels into a mesh, and
 * each new vertex takes its colour from the nearest original vertex.
 *
 * What it costs: detail finer than a voxel (about 4 mm on a chair, 1 mm on a
 * figurine) and crevices no view looks straight into, which get bridged. For a
 * product in AR that is the right trade. The boucle is in the colour, not the mesh.
 */

/**
 * @param prim  a glTF primitive with POSITION (+ optional COLOR_0); replaced in place
 * @param log   progress callback
 * @param directions  views spread over the sphere
 * @param size  depth map resolution per view
 * @param res   voxels along the longest side of the object
 */
export function carveSkin(prim, log, { directions = 64, size = 768, res = null, silhouettes = null, frame = null, refine = 1 } = {}) {
  const posAcc = prim.getAttribute('POSITION');
  const colAcc = prim.getAttribute('COLOR_0');
  const pos = posAcc.getArray();
  const col = colAcc?.getArray() ?? null;
  const tri = prim.getIndices().getArray();
  const n = pos.length / 3;
  // Voxel resolution follows the input's weight. A figurine arrives as 600k
  // triangles and its fins want 2.6 mm voxels (384 along the long side, 15 s); a
  // chair arrives as 6M triangles, has no detail finer than 4 mm, and 384 would
  // spend a minute rasterising it into 16M voxels for nothing.
  if (!res) res = tri.length / 3 < 1.5e6 ? 384 : 256;

  // ---- grid over the vertices the faces still use --------------------------
  const used = new Uint8Array(n);
  for (let i = 0; i < tri.length; i++) used[tri[i]] = 1;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    if (!used[i]) continue;
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], pos[i * 3 + k]); max[k] = Math.max(max[k], pos[i * 3 + k]); }
  }
  const extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const h = Math.max(...extent) / (res - 4);
  const dims = extent.map((e) => Math.ceil(e / h) + 4);
  const origin = min.map((m) => m - 2 * h);
  const [nx, ny, nz] = dims;
  const sx = 1, sy = nx, sz = nx * ny;
  const total = nx * ny * nz;
  // Truncated signed distance, fused across views (positive = outside). A voxel that
  // any view sees in free space, or outside the silhouette, is outside; a voxel within
  // the band around a view's surface takes that view's signed distance; a voxel far
  // behind a view's surface is not touched by that view, which is what keeps a
  // concavity one view looks into from being filled by the views that cannot. A
  // voxel no view ever touches is inside. The zero level set of this is the surface,
  // and the crossings are real distances rather than voxel midpoints, so thin parts
  // survive down to about a voxel instead of the two the blurred binary grid needed.
  const sdfSum = new Float32Array(total);
  const sdfWgt = new Uint16Array(total);
  const TRUNC = 2 * h;

  // ---- depth carving --------------------------------------------------------
  const cx = (min[0] + max[0]) / 2, cy = (min[1] + max[1]) / 2, cz = (min[2] + max[2]) / 2;
  const radius = Math.hypot(extent[0], extent[1], extent[2]) / 2;
  const span = radius * 2.05;
  const px = new Float32Array(n), py = new Float32Array(n), pz = new Float32Array(n);
  const zbuf = new Float32Array(size * size);
  const golden = Math.PI * (3 - Math.sqrt(5));
  const pixel = span / size;

  // One view: its basis, and the depth buffer of the raw mesh seen from it.
  const rasterView = (d) => {
    const y = 1 - (2 * (d + 0.5)) / directions;
    const r = Math.sqrt(1 - y * y);
    const th = golden * d;
    const fwd = [r * Math.cos(th), y, r * Math.sin(th)];
    const helper = Math.abs(fwd[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const right = [
      helper[1] * fwd[2] - helper[2] * fwd[1],
      helper[2] * fwd[0] - helper[0] * fwd[2],
      helper[0] * fwd[1] - helper[1] * fwd[0],
    ];
    const rl = Math.hypot(...right); right[0] /= rl; right[1] /= rl; right[2] /= rl;
    const up = [
      fwd[1] * right[2] - fwd[2] * right[1],
      fwd[2] * right[0] - fwd[0] * right[2],
      fwd[0] * right[1] - fwd[1] * right[0],
    ];
    for (let i = 0; i < n; i++) {
      if (!used[i]) continue;
      const x = pos[i * 3] - cx, yy = pos[i * 3 + 1] - cy, z = pos[i * 3 + 2] - cz;
      px[i] = ((right[0] * x + right[1] * yy + right[2] * z) / span + 0.5) * size;
      py[i] = ((up[0] * x + up[1] * yy + up[2] * z) / span + 0.5) * size;
      pz[i] = fwd[0] * x + fwd[1] * yy + fwd[2] * z;
    }
    zbuf.fill(-Infinity);
    for (let t = 0; t < tri.length; t += 3) {
      const i0 = tri[t], i1 = tri[t + 1], i2 = tri[t + 2];
      const ax = px[i0], ay = py[i0], bx = px[i1], by = py[i1], qx = px[i2], qy = py[i2];
      const ar = (bx - ax) * (qy - ay) - (qx - ax) * (by - ay);
      const lo = Math.max(0, Math.floor(Math.min(ax, bx, qx))), hi = Math.min(size - 1, Math.ceil(Math.max(ax, bx, qx)));
      const vlo = Math.max(0, Math.floor(Math.min(ay, by, qy))), vhi = Math.min(size - 1, Math.ceil(Math.max(ay, by, qy)));
      if (Math.abs(ar) < 1e-9) {
        // a sliver still occupies its pixel: stamp its nearest depth so thin geometry
        // seen edge-on does not become a gap the carve can see through
        const dz = Math.max(pz[i0], pz[i1], pz[i2]);
        for (let yy = vlo; yy <= vhi; yy++) for (let x = lo; x <= hi; x++) { const o = yy * size + x; if (dz > zbuf[o]) zbuf[o] = dz; }
        continue;
      }
      for (let yy = vlo; yy <= vhi; yy++) {
        for (let x = lo; x <= hi; x++) {
          const w0 = ((bx - ax) * (yy + 0.5 - ay) - (x + 0.5 - ax) * (by - ay)) / ar;
          const w1 = ((qx - bx) * (yy + 0.5 - by) - (x + 0.5 - bx) * (qy - by)) / ar;
          const w2 = 1 - w0 - w1;
          // a small tolerance so pixel centres that fall between two triangles are
          // covered by both rather than by neither
          if (w0 < -0.02 || w1 < -0.02 || w2 < -0.02) continue;
          const depth = w1 * pz[i0] + w2 * pz[i1] + w0 * pz[i2];
          const o = yy * size + x;
          if (depth > zbuf[o]) zbuf[o] = depth;
        }
      }
    }

    return { right, up, fwd };
  };

  for (let d = 0; d < directions; d++) {
    const { right, up, fwd } = rasterView(d);
    // fuse this view's depth into the field
    const ox = origin[0] + 0.5 * h - cx, oy = origin[1] + 0.5 * h - cy, oz = origin[2] + 0.5 * h - cz;
    let i = 0;
    for (let iz = 0; iz < nz; iz++) {
      const wz = oz + iz * h;
      for (let iy = 0; iy < ny; iy++) {
        const wy = oy + iy * h;
        // per-row start, then walk x incrementally
        let sxp = ((right[0] * ox + right[1] * wy + right[2] * wz) / span + 0.5) * size;
        let syp = ((up[0] * ox + up[1] * wy + up[2] * wz) / span + 0.5) * size;
        let sdp = fwd[0] * ox + fwd[1] * wy + fwd[2] * wz;
        const dxp = (right[0] * h / span) * size, dyp = (up[0] * h / span) * size, ddp = fwd[0] * h;
        for (let ix = 0; ix < nx; ix++, i++, sxp += dxp, syp += dyp, sdp += ddp) {
          const xi = sxp | 0, yi = syp | 0;
          let d;
          if (xi < 0 || yi < 0 || xi >= size || yi >= size) d = TRUNC;
          else {
            const zb = zbuf[yi * size + xi];
            if (zb === -Infinity) d = TRUNC;
            else {
              d = sdp - zb;                 // positive: in front of the surface
              if (d < -TRUNC) continue;     // far behind it: this view knows nothing
              if (d > TRUNC) d = TRUNC;
            }
          }
          sdfSum[i] += d / TRUNC;
          sdfWgt[i]++;
        }
      }
    }
  }

  // the fused field: mean of what the views said, inside where no view said anything
  let field = new Float32Array(total);
  let inside = 0;
  for (let i = 0; i < total; i++) {
    field[i] = sdfWgt[i] ? sdfSum[i] / sdfWgt[i] : -1;
    if (field[i] < 0) inside++;
  }

  // The photographs' silhouettes, when a first pass aligned them. The depth views
  // only know the reconstruction; a photograph knows the product. Any voxel that
  // projects outside an aligned photograph's subject mask is outside the product, so
  // it is set outside here, hard, not voted: the reconstruction's sixty-four views
  // would otherwise outvote one photograph. The masks are eroded first, because a
  // 70% silhouette fit is wrong at its edge by a few pixels, and carving the real
  // product away is worse than leaving a little invented volume on.
  if (silhouettes?.length && frame) {
    let carved = 0;
    const { centre, scale } = frame;
    for (const sil of silhouettes) {
      const { view, fit, box, mask, width, height } = sil;
      // the frame the silhouette was fitted in: the blank's bounds in view space
      const toBlank = (x, y, z) => [(x - centre[0]) * scale[0], (y - centre[1]) * scale[1], (z - centre[2]) * scale[2]];
      const bounds = sil.bounds;   // { minU, maxU, minV, maxV } of the blank in this view
      const span = (Math.max(bounds.maxU - bounds.minU, bounds.maxV - bounds.minV) || 1) / fit.scale;
      const cu = (bounds.minU + bounds.maxU) / 2, cv = (bounds.minV + bounds.maxV) / 2;
      const erodePx = Math.max(2, Math.round(Math.min(width, height) / 60));
      let i = 0;
      for (let iz = 0; iz < nz; iz++) {
        for (let iy = 0; iy < ny; iy++) {
          for (let ix = 0; ix < nx; ix++, i++) {
            if (field[i] >= 0) continue;   // already outside
            const [bx, by, bz] = toBlank(origin[0] + (ix + 0.5) * h, origin[1] + (iy + 0.5) * h, origin[2] + (iz + 0.5) * h);
            const u = view.right[0] * bx + view.right[1] * by + view.right[2] * bz;
            const v = view.up[0] * bx + view.up[1] * by + view.up[2] * bz;
            const fx = (u - cu) / span + 0.5 + fit.dx;
            const fy = 0.5 - (v - cv) / span + fit.dy;
            const px = Math.round(box.left - box.ox + fx * box.span), py = Math.round(box.top - box.oy + fy * box.span);
            // outside the image or the (eroded) mask: outside the product
            let out = px < erodePx || py < erodePx || px >= width - erodePx || py >= height - erodePx;
            if (!out) {
              out = true;
              // inside only if the whole erosion window is subject
              scan: for (let dy = -erodePx; dy <= erodePx; dy += erodePx) {
                for (let dx = -erodePx; dx <= erodePx; dx += erodePx) {
                  if (!mask[(py + dy) * width + px + dx]) { out = true; break scan; }
                  out = false;
                }
              }
            }
            if (out) { field[i] = 1; carved++; }
          }
        }
      }
    }
    inside -= carved;
    log(`silhouettes of ${silhouettes.length} photograph${silhouettes.length === 1 ? '' : 's'} carved ${carved.toLocaleString()} voxels the reconstruction had invented`);
  }

  // One smoothing pass on the distance field. Views disagree by a wall's thickness
  // wherever the raw mesh is double-walled, and the mean of disagreeing distances is
  // a rough surface: unsmoothed, 13.8% of the hull's corners folded. A separable
  // three-tap pass over real distances is gentler than the two passes the binary
  // grid needed, because the crossings were never quantised to voxel midpoints.
  {
    const tmp = new Float32Array(total);
    for (const [stride, len, outer] of [[sx, nx, [ny, nz, sy, sz]], [sy, ny, [nx, nz, sx, sz]], [sz, nz, [nx, ny, sx, sy]]]) {
      const [la, lb, sa, sb] = outer;
      for (let ia = 0; ia < la; ia++) {
        for (let ib = 0; ib < lb; ib++) {
          const base = ia * sa + ib * sb;
          for (let i = 0; i < len; i++) {
            const c = base + i * stride;
            const lo = i > 0 ? field[c - stride] : field[c];
            const hi = i < len - 1 ? field[c + stride] : field[c];
            tmp[c] = (lo + field[c] + hi) / 3;
          }
        }
      }
      field.set(tmp);
    }
  }
  const ISO = 0;   // the zero level set of the signed distance

  // ---- finer voxels where the surface is ---------------------------------------
  // The coarse grid is sized for the whole object; the detail that matters (a fin,
  // a claw, a tail ridge) is a few voxels wide. Rather than a grid `refine` times
  // finer everywhere (8x the memory and time at 2x), only the blocks that contain the
  // surface get a fine grid, fused from the same depth views, and the surface is
  // extracted from that. Blocks away from the surface keep the coarse field's sign,
  // which is all the extraction needs from them.
  //
  // Off by default, measured: on the figure, 2x gave 15k blocks, 62M fine voxels and
  // a 2.9M-triangle hull in 72 s instead of 34, and after simplification to the
  // sixty-thousand budget the shipped mesh was no sharper and showed more facets.
  // The budget is the limit there, and the relief map carries the detail. It is here
  // for a subject that warrants a bigger budget.
  let fine = null;
  if (refine > 1) {
    const F = refine, B = 8;                       // block = B coarse voxels = B*F fine ones
    const bx = Math.ceil(nx / B), by = Math.ceil(ny / B), bz = Math.ceil(nz / B);
    const isSurface = new Uint8Array(bx * by * bz);
    for (let iz = 0; iz < nz; iz++) for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
      const v = field[ix + iy * sy + iz * sz];
      if (v > -1 && v < 1) {                       // inside the truncation band: near a surface
        for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const X = ((ix + dx * B) / B) | 0, Y = ((iy + dy * B) / B) | 0, Z = ((iz + dz * B) / B) | 0;
          if (X < 0 || Y < 0 || Z < 0 || X >= bx || Y >= by || Z >= bz) continue;
          isSurface[X + Y * bx + Z * bx * by] = 1;
        }
      }
    }
    const blocks = [];
    const blockSlot = new Int32Array(bx * by * bz).fill(-1);
    for (let i = 0; i < isSurface.length; i++) if (isSurface[i]) { blockSlot[i] = blocks.length; blocks.push(i); }
    const BF = B * F, perBlock = BF * BF * BF;
    const fSum = new Float32Array(blocks.length * perBlock);
    const fWgt = new Uint16Array(blocks.length * perBlock);
    const h2 = h / F;

    for (let d = 0; d < directions; d++) {
      const { right, up, fwd } = rasterView(d);
      for (let bIdx = 0; bIdx < blocks.length; bIdx++) {
        const b = blocks[bIdx];
        const BX = b % bx, BY = ((b / bx) | 0) % by, BZ = (b / (bx * by)) | 0;
        const base = bIdx * perBlock;
        for (let fz = 0; fz < BF; fz++) {
          const wz = origin[2] + (BZ * BF + fz + 0.5) * h2 - cz;
          for (let fy = 0; fy < BF; fy++) {
            const wy = origin[1] + (BY * BF + fy + 0.5) * h2 - cy;
            const ox = origin[0] + (BX * BF + 0.5) * h2 - cx;
            let sxp = ((right[0] * ox + right[1] * wy + right[2] * wz) / span + 0.5) * size;
            let syp = ((up[0] * ox + up[1] * wy + up[2] * wz) / span + 0.5) * size;
            let sdp = fwd[0] * ox + fwd[1] * wy + fwd[2] * wz;
            const dxp = (right[0] * h2 / span) * size, dyp = (up[0] * h2 / span) * size, ddp = fwd[0] * h2;
            let o = base + (fz * BF + fy) * BF;
            for (let fx = 0; fx < BF; fx++, o++, sxp += dxp, syp += dyp, sdp += ddp) {
              const xi = sxp | 0, yi = syp | 0;
              let dd;
              if (xi < 0 || yi < 0 || xi >= size || yi >= size) dd = TRUNC;
              else {
                const zb = zbuf[yi * size + xi];
                if (zb === -Infinity) dd = TRUNC;
                else {
                  dd = sdp - zb;
                  if (dd < -TRUNC) continue;
                  if (dd > TRUNC) dd = TRUNC;
                }
              }
              fSum[o] += dd / TRUNC;
              fWgt[o]++;
            }
          }
        }
      }
    }
    // the fine field, with the coarse field as the answer wherever the fine one has
    // none (deep inside, or outside every block)
    const fField = new Float32Array(blocks.length * perBlock);
    const coarseAtFine = (fx, fy, fz) => {
      const ix = Math.min(nx - 1, (fx / F) | 0), iy = Math.min(ny - 1, (fy / F) | 0), iz = Math.min(nz - 1, (fz / F) | 0);
      return field[ix + iy * sy + iz * sz];
    };
    for (let bIdx = 0; bIdx < blocks.length; bIdx++) {
      const b = blocks[bIdx];
      const BX = b % bx, BY = ((b / bx) | 0) % by, BZ = (b / (bx * by)) | 0;
      const base = bIdx * perBlock;
      for (let fz = 0; fz < BF; fz++) for (let fy = 0; fy < BF; fy++) for (let fx = 0; fx < BF; fx++) {
        const o = base + (fz * BF + fy) * BF + fx;
        fField[o] = fWgt[o] ? fSum[o] / fWgt[o] : coarseAtFine(BX * BF + fx, BY * BF + fy, BZ * BF + fz);
      }
    }
    const NX = nx * F, NY = ny * F, NZ = nz * F;
    const sample = (fx, fy, fz) => {
      if (fx < 0 || fy < 0 || fz < 0 || fx >= NX || fy >= NY || fz >= NZ) return 1;
      const slot = blockSlot[((fx / BF) | 0) + ((fy / BF) | 0) * bx + ((fz / BF) | 0) * bx * by];
      if (slot < 0) return coarseAtFine(fx, fy, fz) < 0 ? -1 : 1;
      return fField[slot * perBlock + ((fz % BF) * BF + (fy % BF)) * BF + (fx % BF)];
    };
    // the same one-pass smoothing the coarse field had, three taps per axis
    const smoothed = new Float32Array(fField.length);
    for (let bIdx = 0; bIdx < blocks.length; bIdx++) {
      const b = blocks[bIdx];
      const BX = b % bx, BY = ((b / bx) | 0) % by, BZ = (b / (bx * by)) | 0;
      const base = bIdx * perBlock;
      for (let fz = 0; fz < BF; fz++) for (let fy = 0; fy < BF; fy++) for (let fx = 0; fx < BF; fx++) {
        const X = BX * BF + fx, Y = BY * BF + fy, Z = BZ * BF + fz;
        smoothed[base + (fz * BF + fy) * BF + fx] = (
          sample(X - 1, Y, Z) + sample(X + 1, Y, Z) + sample(X, Y - 1, Z) + sample(X, Y + 1, Z)
          + sample(X, Y, Z - 1) + sample(X, Y, Z + 1) + 2 * sample(X, Y, Z)) / 8;
      }
    }
    fField.set(smoothed);
    fine = { F, B, BF, bx, by, bz, blocks, blockSlot, perBlock, sample, h2, NX, NY, NZ };
    log(`refined ${blocks.length.toLocaleString()} surface blocks to ${(h2 * 1000).toFixed(2)} mm voxels (${(blocks.length * perBlock / 1e6).toFixed(1)}M fine voxels)`);
  }

  // ---- surface nets ---------------------------------------------------------
  // one vertex per cell whose eight corner samples straddle the level, placed at
  // the mean of the interpolated crossing points on the cell's twelve edges
  const verts = [];
  const faces = [];
  if (fine) {
    // the fine grid, walked block by block; cells are keyed sparsely
    const { BF, bx, by, blocks, sample, h2, NX, NY, NZ } = fine;
    const cellKey = (x, y, z) => x + y * NX + z * NX * NY;
    const cellOf = new Map();
    const fv = new Float32Array(8);
    for (const b of blocks) {
      const BX = b % bx, BY = ((b / bx) | 0) % by, BZ = (b / (bx * by)) | 0;
      for (let fz = 0; fz < BF; fz++) for (let fy = 0; fy < BF; fy++) for (let fx = 0; fx < BF; fx++) {
        const X = BX * BF + fx, Y = BY * BF + fy, Z = BZ * BF + fz;
        if (X >= NX - 1 || Y >= NY - 1 || Z >= NZ - 1) continue;
        let mask = 0;
        for (let k = 0; k < 8; k++) {
          fv[k] = sample(X + (k & 1), Y + ((k >> 1) & 1), Z + ((k >> 2) & 1));
          if (fv[k] < ISO) mask |= 1 << k;
        }
        if (mask === 0 || mask === 255) continue;
        let ex = 0, ey = 0, ez = 0, cnt = 0;
        for (let k = 0; k < 8; k++) {
          for (let a = 0; a < 3; a++) {
            if (k & (1 << a)) continue;
            const k2 = k | (1 << a);
            if (((mask >> k) & 1) === ((mask >> k2) & 1)) continue;
            const t = (ISO - fv[k]) / (fv[k2] - fv[k]);
            ex += (k & 1) + (a === 0 ? t : 0);
            ey += ((k >> 1) & 1) + (a === 1 ? t : 0);
            ez += ((k >> 2) & 1) + (a === 2 ? t : 0);
            cnt++;
          }
        }
        cellOf.set(cellKey(X, Y, Z), verts.length / 3);
        verts.push(origin[0] + (X + 0.5 + ex / cnt) * h2, origin[1] + (Y + 0.5 + ey / cnt) * h2, origin[2] + (Z + 0.5 + ez / cnt) * h2);
      }
    }
    const step = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (const [key, v0] of cellOf) {
      const X = key % NX, Y = ((key / NX) | 0) % NY, Z = (key / (NX * NY)) | 0;
      if (X < 1 || Y < 1 || Z < 1) continue;
      const s0 = sample(X, Y, Z) < ISO ? 1 : 0;
      for (let a = 0; a < 3; a++) {
        const s1 = sample(X + step[a][0], Y + step[a][1], Z + step[a][2]) < ISO ? 1 : 0;
        if (s0 === s1) continue;
        const u = step[(a + 1) % 3], w = step[(a + 2) % 3];
        const v1 = cellOf.get(cellKey(X - u[0], Y - u[1], Z - u[2]));
        const v2 = cellOf.get(cellKey(X - u[0] - w[0], Y - u[1] - w[1], Z - u[2] - w[2]));
        const v3 = cellOf.get(cellKey(X - w[0], Y - w[1], Z - w[2]));
        if (v1 === undefined || v2 === undefined || v3 === undefined) continue;
        if (s0) faces.push(v0, v1, v2, v0, v2, v3);
        else faces.push(v0, v3, v2, v0, v2, v1);
      }
    }
  }
  const cellIndex = fine ? null : new Int32Array(total).fill(-1);
  if (!fine) {
  const sampleAt = (x, y, z) => field[x + y * sy + z * sz];
  const occ = new Uint8Array(total);
  for (let i = 0; i < total; i++) occ[i] = field[i] < ISO ? 1 : 0;   // inside = negative
  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      for (let x = 0; x < nx - 1; x++) {
        let mask = 0;
        const fv = new Float32Array(8);
        for (let k = 0; k < 8; k++) {
          fv[k] = sampleAt(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1));
          if (fv[k] < ISO) mask |= 1 << k;
        }
        if (mask === 0 || mask === 255) continue;
        let ex = 0, ey = 0, ez = 0, cnt = 0;
        for (let k = 0; k < 8; k++) {
          for (let a = 0; a < 3; a++) {
            if (k & (1 << a)) continue;                 // edge goes from k to k | (1<<a)
            const k2 = k | (1 << a);
            if (((mask >> k) & 1) === ((mask >> k2) & 1)) continue;
            // interpolated crossing along the edge, in cell units
            const t = (ISO - fv[k]) / (fv[k2] - fv[k]);
            ex += (k & 1) + (a === 0 ? t : 0);
            ey += ((k >> 1) & 1) + (a === 1 ? t : 0);
            ez += ((k >> 2) & 1) + (a === 2 ? t : 0);
            cnt++;
          }
        }
        cellIndex[x + y * sy + z * sz] = verts.length / 3;
        // samples sit at voxel centres: sample (x) is at origin + (x + 0.5) h
        verts.push(
          origin[0] + (x + 0.5 + ex / cnt) * h,
          origin[1] + (y + 0.5 + ey / cnt) * h,
          origin[2] + (z + 0.5 + ez / cnt) * h,
        );
      }
    }
  }

  const strides = [sx, sy, sz];
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const c = x + y * sy + z * sz;
        const v0 = cellIndex[c];
        if (v0 < 0) continue;
        const s0 = occ[c];
        for (let a = 0; a < 3; a++) {
          const s1 = occ[c + strides[a]];
          if (s0 === s1) continue;
          const du = strides[(a + 1) % 3], dv = strides[(a + 2) % 3];
          const v1 = cellIndex[c - du], v2 = cellIndex[c - du - dv], v3 = cellIndex[c - dv];
          if (v1 < 0 || v2 < 0 || v3 < 0) continue;
          if (s0) faces.push(v0, v1, v2, v0, v2, v3);
          else faces.push(v0, v3, v2, v0, v2, v1);
        }
      }
    }
  }

  }   // end of the coarse-grid extraction

  // winding: make the signed volume positive, so normals point out
  let vol = 0;
  for (let t = 0; t < faces.length; t += 3) {
    const a = faces[t] * 3, b = faces[t + 1] * 3, c = faces[t + 2] * 3;
    vol += verts[a] * (verts[b + 1] * verts[c + 2] - verts[b + 2] * verts[c + 1])
      - verts[a + 1] * (verts[b] * verts[c + 2] - verts[b + 2] * verts[c])
      + verts[a + 2] * (verts[b] * verts[c + 1] - verts[b + 1] * verts[c]);
  }
  if (vol < 0) for (let t = 0; t < faces.length; t += 3) { const tmp = faces[t + 1]; faces[t + 1] = faces[t + 2]; faces[t + 2] = tmp; }

  // ---- how much of the surface is paper-thin ---------------------------------
  // A reconstruction that went wrong often goes wrong as a sheet: the three-quarter
  // build of the Shin Godzilla figure turned the tail into a flat plate that still
  // scored 77% on silhouette. Walk inward from each vertex along its normal; if the
  // field is outside again within 2.5 voxels, the surface has no body behind it.
  // Fins and a tail are thin by right, so this is a share to warn on, not a verdict.
  let thinShare = 0;
  {
    const nv = verts.length / 3;
    const vn = new Float64Array(nv * 3);
    for (let t = 0; t < faces.length; t += 3) {
      const a = faces[t], b = faces[t + 1], c = faces[t + 2];
      const e1 = [verts[b * 3] - verts[a * 3], verts[b * 3 + 1] - verts[a * 3 + 1], verts[b * 3 + 2] - verts[a * 3 + 2]];
      const e2 = [verts[c * 3] - verts[a * 3], verts[c * 3 + 1] - verts[a * 3 + 1], verts[c * 3 + 2] - verts[a * 3 + 2]];
      const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
      for (const v of [a, b, c]) { vn[v * 3] += nx; vn[v * 3 + 1] += ny; vn[v * 3 + 2] += nz; }
    }
    let thin = 0;
    const at = (x, y, z) => {
      const ix = Math.round((x - origin[0]) / h - 0.5), iy = Math.round((y - origin[1]) / h - 0.5), iz = Math.round((z - origin[2]) / h - 0.5);
      if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return 1;
      return field[ix + iy * sy + iz * sz];
    };
    for (let v = 0; v < nv; v++) {
      const l = Math.hypot(vn[v * 3], vn[v * 3 + 1], vn[v * 3 + 2]) || 1;
      const dx = vn[v * 3] / l, dy = vn[v * 3 + 1] / l, dz = vn[v * 3 + 2] / l;
      for (const k of [1.5, 2.5]) {
        if (at(verts[v * 3] - dx * k * h, verts[v * 3 + 1] - dy * k * h, verts[v * 3 + 2] - dz * k * h) > 0) { thin++; break; }
      }
    }
    thinShare = nv ? thin / nv : 0;
  }

  // ---- colour from the nearest original vertex ----------------------------
  let newCol = null;
  if (col) {
    const cell = 2 * h;
    const grid = new Map();
    const key = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
    for (let i = 0; i < n; i++) {
      if (!used[i]) continue;
      const k = key(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      let l = grid.get(k); if (!l) grid.set(k, l = []); l.push(i);
    }
    const m = verts.length / 3;
    newCol = new Float32Array(m * 3);
    let mr = 0, mg = 0, mb = 0, mc = 0;
    for (let i = 0; i < n; i++) if (used[i]) { mr += col[i * 3]; mg += col[i * 3 + 1]; mb += col[i * 3 + 2]; mc++; }
    const fallback = [mr / mc, mg / mc, mb / mc];
    for (let v = 0; v < m; v++) {
      const x = verts[v * 3], y = verts[v * 3 + 1], z = verts[v * 3 + 2];
      const gx = Math.floor(x / cell), gy = Math.floor(y / cell), gz = Math.floor(z / cell);
      let best = -1, bestD = Infinity;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const l = grid.get(`${gx + dx},${gy + dy},${gz + dz}`); if (!l) continue;
        for (const j of l) {
          const dd = (pos[j * 3] - x) ** 2 + (pos[j * 3 + 1] - y) ** 2 + (pos[j * 3 + 2] - z) ** 2;
          if (dd < bestD) { bestD = dd; best = j; }
        }
      }
      if (best >= 0) { newCol[v * 3] = col[best * 3]; newCol[v * 3 + 1] = col[best * 3 + 1]; newCol[v * 3 + 2] = col[best * 3 + 2]; }
      else { newCol[v * 3] = fallback[0]; newCol[v * 3 + 1] = fallback[1]; newCol[v * 3 + 2] = fallback[2]; }
    }
  }

  // ---- replace the primitive's data ----------------------------------------
  for (const sem of prim.listSemantics()) {
    if (sem !== 'POSITION' && sem !== 'COLOR_0') prim.setAttribute(sem, null);
  }
  posAcc.setArray(new Float32Array(verts));
  if (colAcc) colAcc.setArray(newCol);
  prim.getIndices().setArray(new Uint32Array(faces));

  log(`carved the outer skin: ${directions} views, ${(h * 1000).toFixed(1)} mm voxels, ${inside.toLocaleString()} of ${total.toLocaleString()} inside, ${(verts.length / 3).toLocaleString()} vertices, ${(faces.length / 3).toLocaleString()} triangles, ${(thinShare * 100).toFixed(0)}% of the surface under 2.5 voxels thick`);
  return { voxel: h, directions, triangles: faces.length / 3, thinShare };
}
