// Does the multi-view checkpoint actually improve on the single-view one?
//
// Controlled A/B: identical inputs and settings, only the weights differ. The second
// view is a mirrored front, not a real photograph, so this measures the CHECKPOINT
// effect in isolation rather than multi-view's full value on honest captures. The
// baseline numbers it must beat, measured earlier on this same input through the
// single-view weights: 1 view 0.4347, 2 views 0.4259 (the second view made it WORSE,
// because the weights had never seen the layout).
//
// Usage: node tools/ab-mv.mjs
import { readFile, mkdir } from 'node:fs/promises';
import { reconstruct, reconstructViews, readiness } from '../lib/recon.mjs';
import { makeBlank } from '../lib/blank.mjs';
import { findView } from '../lib/align.mjs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const run = promisify(execFile);

const SRC = 'projects/ed5445eb-c09';
const ready = await readiness();
console.log('mv checkpoint:', ready.mvCheckpoint || 'NOT INSTALLED');
if (!ready.mvCheckpoint) process.exit(1);

const front = await readFile(`${SRC}/matted-front.png`);
const right = await readFile(`${SRC}/matted-right.png`);
await run('python', ['lib/photomask.py', `${SRC}/matted-front.png`, `${SRC}/mask.bin`, '192']).catch(async () => {
  // older projects stored the plain matte under a different name
  await run('python', ['lib/photomask.py', `${SRC}/matted.png`, `${SRC}/mask.bin`, '192']);
});
const mask = new Uint8Array(await readFile(`${SRC}/mask.bin`));

async function score(dir) {
  const { NodeIO } = await import('@gltf-transform/core');
  const doc = await new NodeIO().read(`${dir}/blank.glb`);
  const p = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
  const { best } = findView(p.getAttribute('POSITION').getArray(), p.getIndices().getArray(), mask, 192, 10);
  return best;
}

const runs = [
  ['1 view, single-view weights', () => reconstruct(front, 'abtmp/one.glb',
    { octreeResolution: 256, steps: 30, onProgress: () => {} })],
  ['2 views, mv weights', () => reconstructViews({ front, right }, 'abtmp/two.glb',
    { octreeResolution: 256, steps: 30, onProgress: () => {} })],
];

await mkdir('abtmp', { recursive: true });
for (const [name, go] of runs) {
  const t0 = Date.now();
  await go();
  const src = name.startsWith('1') ? 'abtmp/one.glb' : 'abtmp/two.glb';
  await makeBlank(src, 'abtmp', { targetHeight: 0.78, onProgress: () => {} });
  const best = await score('abtmp');
  console.log(`${name.padEnd(30)} IoU ${best.score.toFixed(4)}  (${best.view.name})  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
console.log('baselines through single-view weights: 1v 0.4347, 2v 0.4259');
