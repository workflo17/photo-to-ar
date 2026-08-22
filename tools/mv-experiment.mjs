// Does multi-view reconstruction actually fix the invented back?
//
// The one experiment the mirrored-front A/B could not run. The inputs are four
// renders of a known 3D mesh, consistent by construction, and the score is silhouette
// IoU against that same mesh from all four sides. A single-view reconstruction should
// hold up at the front and fall apart at the sides and back, because it never saw
// them; if four views are worth shipping, the four-view run has to close that gap.
//
// Scoring: one global yaw offset aligns each reconstruction to the ground truth (both
// stand upright after makeBlank), chosen to maximise the MEAN IoU across all four
// sides. Per-side numbers are reported at that alignment, so a model cannot win the
// front by spinning its best face toward every camera.
//
// Usage: node tools/mv-experiment.mjs
import { readFile, mkdir } from 'node:fs/promises';
import { NodeIO } from '@gltf-transform/core';
import { reconstruct, reconstructViews, readiness } from '../lib/recon.mjs';
import { makeBlank } from '../lib/blank.mjs';
import { silhouette, iou } from '../lib/align.mjs';

const GT = 'projects/665477d4-c02';
const ready = await readiness();
if (!ready.mvCheckpoint) { console.error('mv checkpoint missing'); process.exit(1); }

const viewAt = (deg) => {
  const r = (deg * Math.PI) / 180;
  return { right: [Math.cos(r), 0, -Math.sin(r)], up: [0, 1, 0] };
};
const SIDES = [0, 90, 180, 270];

async function meshArrays(path) {
  const doc = await new NodeIO().read(path);
  const p = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
  return { pos: p.getAttribute('POSITION').getArray(), idx: p.getIndices().getArray() };
}

const gt = await meshArrays(`${GT}/blank.glb`);
const gtMasks = SIDES.map((d) => silhouette(gt.pos, gt.idx, viewAt(d), 192));

function scoreAgainstGT(recon) {
  let best = { mean: -1, offset: 0, per: [] };
  for (let off = 0; off < 360; off += 10) {
    const per = SIDES.map((d, k) => iou(silhouette(recon.pos, recon.idx, viewAt(d + off), 192), gtMasks[k]));
    const mean = per.reduce((a, b) => a + b, 0) / per.length;
    if (mean > best.mean) best = { mean, offset: off, per };
  }
  return best;
}

const front = await readFile('gtviews/front.png');
const views = {
  front,
  right: await readFile('gtviews/right.png'),
  back: await readFile('gtviews/back.png'),
  left: await readFile('gtviews/left.png'),
};

await mkdir('abtmp', { recursive: true });
const RUNS = [
  ['1 view', () => reconstruct(front, 'abtmp/mv1.glb', { octreeResolution: 256, steps: 30, onProgress: () => {} }), 'abtmp/mv1.glb'],
  ['4 views (mv weights)', () => reconstructViews(views, 'abtmp/mv4.glb', { octreeResolution: 256, steps: 30, onProgress: () => {} }), 'abtmp/mv4.glb'],
];

console.log('side:                     front   right    back    left    MEAN');
for (const [name, go, out] of RUNS) {
  const t0 = Date.now();
  await go();
  await makeBlank(out, 'abtmp', { targetHeight: 0.78, onProgress: () => {} });
  const recon = await meshArrays('abtmp/blank.glb');
  const s = scoreAgainstGT(recon);
  console.log(`${name.padEnd(24)} ${s.per.map((v) => v.toFixed(3)).join('   ')}   ${s.mean.toFixed(3)}  (yaw offset ${s.offset}, ${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}
