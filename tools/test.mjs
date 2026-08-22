// End-to-end check of everything except the GPU reconstruction, which is too slow
// and too stateful to run on every change. Drives the real studio in a real browser:
// loads a project, paints parts, bakes, and confirms both AR files come back.
//
// Usage: node tools/test.mjs [baseUrl]
import { spawn } from 'node:child_process';
import { writeFileSync, existsSync } from 'node:fs';

const BASE = process.argv[2] || 'http://localhost:5140';
const EDGE = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const PORT = 9241;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let pass = 0, fail = 0;
const check = (name, ok, note = '') => {
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${name}${note ? `  (${note})` : ''}`);
  ok ? pass++ : fail++;
};

/* ---- server side, no browser needed ---- */
const status = await (await fetch(`${BASE}/api/status`)).json();
check('status reports the GPU', status.gpu === true, status.reason || '');
check('material catalogue is served', status.materials?.length >= 9, `${status.materials?.length} materials`);

const projects = await (await fetch(`${BASE}/api/projects`)).json();
check('at least one project exists', projects.length > 0);
const id = projects[0]?.id;
if (!id) { console.log('\nno project to test against'); process.exit(1); }

for (const [file, type] of [['blank.glb', 'model/gltf-binary'], ['regions.json', 'application/json'], ['look.json', 'application/json']]) {
  const r = await fetch(`${BASE}/p/${id}/${file}`);
  check(`serves ${file}`, r.ok && r.headers.get('content-type').startsWith(type.split(';')[0]));
}

// path traversal must not escape the projects directory
const eviltry = await fetch(`${BASE}/p/${id}/..%2f..%2fserver.mjs`);
check('rejects path traversal', !eviltry.ok || !(await eviltry.text()).includes('createServer'));

/* ---- the browser half ---- */
const proc = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${PORT}`,
  '--remote-allow-origins=*', `--user-data-dir=${process.env.TEMP || '.'}/edge-p2ar-test`,
  '--no-first-run', '--window-size=1600,1000', '--hide-scrollbars',
  '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--enable-webgl', 'about:blank'],
  { stdio: 'ignore' });

let targets = null;
for (let i = 0; i < 60 && !targets; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); } catch { await sleep(400); }
}
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let mid = 0; const pending = new Map(); const errors = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') {
    errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  }
};
const send = (method, params = {}) => new Promise((res) => {
  const i = ++mid; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (e) => {
  const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true });
  // Runtime.evaluate reports a throw as exceptionDetails on the reply rather than
  // as a Runtime.exceptionThrown event, so ignoring it makes a failed step look
  // like a step that simply did nothing.
  const ex = r.result?.exceptionDetails;
  if (ex) errors.push(ex.exception?.description || ex.text);
  return r.result?.result?.value;
};

await send('Page.enable'); await send('Runtime.enable');
await send('Page.navigate', { url: BASE });
await sleep(3000);
await evaluate(`document.querySelector('.recent').click()`);
// Poll for the studio instead of guessing. A finer reconstruction means a bigger
// blank.glb, so any fixed wait that works today breaks the next time quality goes up.
for (let i = 0; i < 60; i++) {
  if (await evaluate(`document.querySelectorAll('.region').length > 0
    && document.querySelector('#view').width > 100`)) break;
  await sleep(500);
}

const loaded = JSON.parse(await evaluate(`JSON.stringify({
  regions: document.querySelectorAll('.region').length,
  canvas: document.querySelector('#view').width,
  uniqueLabels: new Set([...document.querySelectorAll('.region span')].map(e=>e.textContent)).size,
})`));
check('studio loaded the blank', loaded.regions > 0, `${loaded.regions} regions`);
check('canvas has a real size', loaded.canvas > 100, `${loaded.canvas}px`);
check('region labels are unique', loaded.uniqueLabels === loaded.regions,
  `${loaded.uniqueLabels}/${loaded.regions}`);

// paint: three different materials on three different parts
const paintResult = await evaluate(`(()=>{
  const regions = [...document.querySelectorAll('.region')];
  const mats = [...document.querySelectorAll('.mat')];
  // indices relative to what the segmentation actually found: the part count moves
  // with the reconstruction quality, so anything hardcoded here goes stale
  const pick = (f) => regions[Math.min(regions.length - 1, Math.floor(regions.length * f))];
  pick(0).click(); mats.find(m=>m.dataset.id==='ink-velvet').click();
  document.querySelector('#tint').value = '#2d3a7a';
  document.querySelector('#tint').dispatchEvent(new Event('input'));
  pick(0.34).click(); mats.find(m=>m.dataset.id==='metal').click();
  pick(0.67).click(); mats.find(m=>m.dataset.id==='camel-boucle').click();
  document.querySelector('#tile').value = '2';
  document.querySelector('#tile').dispatchEvent(new Event('input'));
  return 'painted';
})()`);
check('paint script ran to completion', paintResult === 'painted', String(paintResult));
await sleep(2500);

const painted = JSON.parse(await evaluate(`JSON.stringify({
  swatches: new Set([...document.querySelectorAll('.region i')].map(e=>e.style.background)).size,
  exportReady: document.querySelector('#export').classList.contains('ready'),
})`));
check('painting changed more than one part', painted.swatches > 1, `${painted.swatches} distinct swatches`);
check('export button armed', painted.exportReady);

const shot1 = await send('Page.captureScreenshot', { format: 'png' });
if (shot1.result?.data) writeFileSync('studio-painted.png', Buffer.from(shot1.result.data, 'base64'));

// bake, and wait for the server to finish both formats
await evaluate(`document.querySelector('#export').click()`);
// Poll rather than sleep. The first bake of a session pays for Python importing
// OpenUSD cold, which can take twenty seconds; every later one takes about one. A
// fixed wait is therefore either flaky or slow, and this is neither.
for (let i = 0; i < 90; i++) {
  const busy = await evaluate(`document.querySelector('#export').disabled`);
  if (!busy && await evaluate(`/material|failed/i.test(document.querySelector('#export-note').textContent)`)) break;
  await sleep(1000);
}
const baked = JSON.parse(await evaluate(`JSON.stringify({
  note: document.querySelector('#export-note').textContent,
  label: document.querySelector('#export').textContent,
  arShown: !document.querySelector('#ar-row').hidden,
})`));
check('bake reported success', /material/.test(baked.note) && !/failed/i.test(baked.note), baked.note);
check('AR link revealed', baked.arShown);

/* ---- the AR handoff ---- */
for (const [file, type] of [['model.glb', 'model/gltf-binary'], ['model.usdz', 'model/vnd.usdz+zip']]) {
  const r = await fetch(`${BASE}/p/${id}/${file}`);
  check(`bake produced ${file}`, r.ok, `${(Number(r.headers.get('content-length')) / 1e6).toFixed(2)} MB`);
  check(`${file} has the right content type`, r.headers.get('content-type') === type,
    r.headers.get('content-type'));
}
/* ---- the AR file contents, not just their headers -----------------------------
 * Each of these corresponds to a bug that shipped silently: a USDZ that is wrong in
 * these ways still opens, still validates and still renders, just in the wrong colour
 * or the wrong lighting, on iOS only.
 */
{
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const run = promisify(execFile);
  const dir = `projects/${id}`;
  try {
    const { stdout } = await run('python', ['tools/usdz_facts.py', `${dir}/model.usdz`],
      { maxBuffer: 1 << 24 });
    const f = JSON.parse(stdout);
    check('USDZ carries no authored displayColor', f.meshesWithAuthoredDisplayColor === 0,
      `${f.meshesWithAuthoredDisplayColor} meshes (AR Quick Look ignores it, so the tint would be lost)`);
    check('USDZ is at metric scale, Y up', f.metersPerUnit === 1 && /y/i.test(f.upAxis),
      `${f.metersPerUnit} m/unit, ${f.upAxis}`);
    if (f.normTexScales.length) {
      const ok = f.normTexScales.every((s) => s && s[0] === 2 && s[1] === 2 && s[2] === 2)
        && f.normTexBiases.every((b) => b && b[0] === -1 && b[1] === -1 && b[2] === -1);
      check('USDZ normal maps carry the 8-bit scale/bias remap', ok,
        `scale ${JSON.stringify(f.normTexScales[0])} bias ${JSON.stringify(f.normTexBiases[0])}`);
      check('USDZ normal maps are raw, not colour managed',
        f.normTexColorSpaces.every((c) => c === 'raw'), JSON.stringify(f.normTexColorSpaces));
      check('USDZ tint reaches iOS via the sampler scale',
        f.baseTexScales.length > 0 && f.baseTexScales.every((s) => s && s.length === 4),
        JSON.stringify(f.baseTexScales[0]));
    }
  } catch (e) {
    check('USDZ contents are readable', false, (e.stderr || e.message).slice(0, 120));
  }

  const { NodeIO } = await import('@gltf-transform/core');
  const { KHRMaterialsSheen } = await import('@gltf-transform/extensions');
  // register it or the read logs a warning about an extension the file legitimately uses
  const doc = await new NodeIO().registerExtensions([KHRMaterialsSheen]).read(`${dir}/model.glb`);
  const prims = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives());
  check('glTF authors a tangent basis on every primitive',
    prims.length > 0 && prims.every((p) => p.getAttribute('TANGENT')),
    `${prims.filter((p) => p.getAttribute('TANGENT')).length}/${prims.length} primitives`);
}

/* ---- the photo-colour path, on any project that has one ---------------------- */
{
  const withColour = projects.find((p) => p.hasColour);
  if (!withColour) {
    console.log('  skip  photo-colour path (no coloured project on disk)');
  } else {
    const d = `projects/${withColour.id}`;
    const { stat } = await import('node:fs/promises');
    const jpg = await stat(`${d}/photo.jpg`).catch(() => null);
    const bin = await stat(`${d}/atlas.bin`).catch(() => null);
    check('a coloured blank has a baked photo map', !!jpg && jpg.size > 20000,
      jpg ? `${(jpg.size / 1024).toFixed(0)} KB` : 'missing');
    check('a coloured blank has atlas UVs', !!bin && bin.size % 24 === 0,
      bin ? `${bin.size / 24} triangles` : 'missing');
    const cat = (await (await fetch(`${BASE}/api/status`)).json()).materials;
    check('the catalogue offers the photo material', cat.some((m) => m.kind === 'photo'));

    // The atlas must be the right way up in glTF's convention (v grows DOWN the
    // image). It was upside down for a long time: every glTF viewer then showed each
    // face the opposite side's colour, and a chair wore its legs across the top of
    // its back, while the studio and the USDZ, reading the other convention, looked
    // fine. Read photo.jpg at the atlas UVs of many vertices and compare with the
    // vertex colour there: a flip makes the two disagree on most of them.
    {
      const { NodeIO } = await import('@gltf-transform/core');
      const sharp = (await import('sharp')).default;
      const doc = await new NodeIO().read(`${d}/blank.glb`);
      const prim = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
      const col = prim.getAttribute('COLOR_0').getArray(), idx = prim.getIndices().getArray();
      const raw = await (await import('node:fs/promises')).readFile(`${d}/atlas.bin`);
      const uv = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
      const img = await sharp(`${d}/photo-splat.jpg`).raw().toBuffer({ resolveWithObject: true }).catch(() => null)
        || await sharp(`${d}/photo.jpg`).raw().toBuffer({ resolveWithObject: true });
      const S = img.info.width, ch = img.info.channels;
      const srgb = (c) => 255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
      let agree = 0, n = 0;
      for (let t = 0; t < idx.length / 3; t += 97) {
        const v = idx[t * 3];
        const x = Math.round(uv[t * 6] * (S - 1)), y = Math.round(uv[t * 6 + 1] * (S - 1));
        const o = (y * S + x) * ch;
        const dl = Math.abs((img.data[o] + img.data[o + 1] + img.data[o + 2]) / 3
          - (srgb(col[v * 3]) + srgb(col[v * 3 + 1]) + srgb(col[v * 3 + 2])) / 3);
        if (dl < 40) agree++;
        n++;
      }
      check('the colour atlas is the right way up for glTF', agree / n > 0.6, `${agree}/${n} sampled vertices match their texel`);
    }

    // The relief map: baked from the raw mesh onto the same atlas, carried by the
    // photo material in the glTF as a normal texture. A carved blank always has one.
    const withDetail = projects.find((p) => p.hasDetail);
    if (!withDetail) {
      console.log('  skip  detail map (no carved project on disk)');
    } else {
      const dd = `projects/${withDetail.id}`;
      const png = await stat(`${dd}/detail.png`).catch(() => null);
      check('a carved blank has a detail normal map', !!png && png.size > 20000,
        png ? `${(png.size / 1024).toFixed(0)} KB` : 'missing');
      const { readFile, writeFile } = await import('node:fs/promises');
      const regions = JSON.parse(await readFile(`${dd}/regions.json`, 'utf8'));
      const { emptyLook, setAll } = await import('../lib/look.mjs');
      const look = setAll(emptyLook(regions.regions.length), { materialId: 'photo', tint: '#ffffff' });
      const baked = await (await fetch(`${BASE}/api/projects/${withDetail.id}/bake`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(look),
      })).json();
      const { NodeIO } = await import('@gltf-transform/core');
      const doc = await new NodeIO().read(`${dd}/model.glb`);
      const mats = doc.getRoot().listMaterials();
      check('the photo material ships its relief as a normal texture',
        baked.materials === 1 && mats.length === 1 && !!mats[0].getNormalTexture(),
        `${mats.length} material${mats.length === 1 ? '' : 's'}, normal texture ${mats[0]?.getNormalTexture() ? 'present' : 'absent'}`);
      const { execFile } = await import('node:child_process');
      const { promisify } = await import('node:util');
      const { stdout: factsOut } = await promisify(execFile)('python', ['tools/usdz_facts.py', `${dd}/model.usdz`], { maxBuffer: 1 << 24 }).catch(() => ({ stdout: '' }));
      const facts = factsOut ? JSON.parse(factsOut) : null;
      const normFiles = facts?.normTexFiles || [];
      check('the USDZ carries the relief too, unpacked for 8-bit',
        normFiles.some((f) => /detail\.png@?$/.test(f || '')) && facts.normTexScales.every((sc) => sc && sc[0] === 2),
        facts ? `${normFiles.join(', ') || 'no normal texture'}; scales ${JSON.stringify(facts.normTexScales)}` : 'could not read the USDZ');
    }
  }
}

/* ---- delivery: compression, revalidation, forced download --------------------- */
{
  const gz = await fetch(`${BASE}/vendor/model-viewer/model-viewer.min.js`,
    { headers: { 'accept-encoding': 'gzip' } });
  // node's fetch decompresses transparently, so assert on the header, not the bytes
  check('the model-viewer bundle is gzipped on the wire',
    gz.headers.get('content-encoding') === 'gzip',
    `content-encoding: ${gz.headers.get('content-encoding')}`);

  const first = await fetch(`${BASE}/p/${id}/model.glb`);
  const tag = first.headers.get('etag');
  const again = await fetch(`${BASE}/p/${id}/model.glb`, { headers: { 'if-none-match': tag } });
  check('a repeat AR launch revalidates with 304', again.status === 304, `status ${again.status}`);

  const dl = await fetch(`${BASE}/p/${id}/model.usdz?dl=1`);
  check('?dl=1 forces a download instead of AR Quick Look',
    /attachment/.test(dl.headers.get('content-disposition') || ''),
    dl.headers.get('content-disposition'));
}

/* ---- the secure listener Android AR depends on -------------------------------- */
{
  const https = await import('node:https');
  const port = new URL(BASE).port ? Number(new URL(BASE).port) + 1 : 5141;
  const status = await new Promise((resolve) => {
    const req = https.request({ host: 'localhost', port, path: `/p/${id}/model.glb`, method: 'HEAD',
      rejectUnauthorized: false }, (res) => { resolve(res.statusCode); res.resume(); });
    req.on('error', () => resolve(0));
    req.end();
  });
  check('the https listener serves the model (Android AR needs it)', status === 200, `https :${port} -> ${status}`);
}

const arPage = await (await fetch(`${BASE}/ar/${id}`)).text();
check('the AR page knows the secure port', arPage.includes(':5141') && !arPage.includes('__HTTPS_PORT__'));
check('AR page names both sources',
  arPage.includes(`/p/${id}/model.glb`) && arPage.includes(`ios-src="/p/${id}/model.usdz"`));
check('AR page fixes the scale', arPage.includes('ar-scale="fixed"'));

await send('Page.navigate', { url: `${BASE}/ar/${id}` });
await sleep(7000);
const shot2 = await send('Page.captureScreenshot', { format: 'png' });
if (shot2.result?.data) writeFileSync('ar-page.png', Buffer.from(shot2.result.data, 'base64'));

check('no uncaught exceptions in the page', errors.length === 0, errors.slice(0, 2).join(' | '));

proc.kill();
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
