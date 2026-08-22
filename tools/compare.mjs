// Does changing the reconstruction settings actually make it better?
//
// Runs the same photograph through several configurations and reports what each one
// produced, so the quality argument is settled by measurement rather than by which
// setting sounds more impressive. Renders a turntable frame of each for eyeballing.
//
// Usage: node tools/compare.mjs <photo.png>
import { mkdir, writeFile, readFile, stat } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { join } from 'node:path';
import { reconstruct } from '../lib/recon.mjs';
import { makeBlank } from '../lib/blank.mjs';

const run = promisify(execFile);
const PHOTO = process.argv[2];
const OUT = 'compare';
await mkdir(OUT, { recursive: true });

const VARIANTS = [
  { id: 'a-current', algorithm: 'basic', octree: 256, matte: false, note: 'what shipped' },
  { id: 'b-surfacenet', algorithm: 'surface net', octree: 256, matte: false, note: 'better isosurface' },
  { id: 'c-octree512', algorithm: 'surface net', octree: 512, matte: false, note: 'and a finer grid' },
  { id: 'd-matted', algorithm: 'surface net', octree: 512, matte: true, note: 'and no backdrop' },
];

// matte once, reuse
const matted = join(OUT, 'matted.png');
let mattedOk = false;
try {
  const { stdout } = await run('python', ['lib/matte.py', PHOTO, matted]);
  console.log('matte:', stdout.trim());
  mattedOk = true;
} catch (e) {
  console.log('matte failed or found no backdrop:', (e.stdout || e.message).trim());
}

const rows = [];
for (const v of VARIANTS) {
  if (v.matte && !mattedOk) { console.log(`skipping ${v.id}: no matte`); continue; }
  const dir = join(OUT, v.id);
  await mkdir(dir, { recursive: true });
  const image = await readFile(v.matte ? matted : PHOTO);
  const t0 = Date.now();
  console.log(`\n=== ${v.id}: ${v.note}`);
  try {
    const r = await reconstruct(image, join(dir, 'raw.glb'), {
      algorithm: v.algorithm, octreeResolution: v.octree,
      onProgress: (p) => process.stdout.write(`\r    ${p.detail.padEnd(40)}`),
    });
    const secs = (Date.now() - t0) / 1000;
    const meta = await makeBlank(join(dir, 'raw.glb'), dir, { targetHeight: 0.78, onProgress: () => {} });
    rows.push({
      id: v.id, note: v.note, secs: secs.toFixed(0),
      rawMB: (r.bytes / 1e6).toFixed(1),
      rawTris: meta.rawTriangles,
      floorCut: meta.floorRemoved ?? null,
      regions: meta.regions,
      dims: `${(meta.dimensions.width * 100).toFixed(0)}x${(meta.dimensions.depth * 100).toFixed(0)}`,
    });
    console.log(`\n    ${meta.rawTriangles.toLocaleString()} raw triangles, ${meta.regions} regions, ${secs.toFixed(0)}s`);
  } catch (e) {
    console.log(`\n    FAILED: ${e.message}`);
    rows.push({ id: v.id, note: v.note, failed: e.message.slice(0, 60) });
  }
}

console.log('\n\n  variant        raw tris    raw MB   regions   footprint   secs   note');
for (const r of rows) {
  if (r.failed) { console.log(`  ${r.id.padEnd(14)} FAILED: ${r.failed}`); continue; }
  console.log(`  ${r.id.padEnd(14)} ${String(r.rawTris).padStart(9)} ${String(r.rawMB).padStart(8)} `
    + `${String(r.regions).padStart(9)} ${r.dims.padStart(11)} ${String(r.secs).padStart(6)}   ${r.note}`);
}
await writeFile(join(OUT, 'results.json'), JSON.stringify(rows, null, 2));
