// Package one finished project as a page you can hand to a client.
//
// The studio and the AR page live on this machine's LAN, which is right for making
// the thing and wrong for showing it to someone. This writes a folder that any static
// host will serve: the product in a viewer, a "view in your room" button that opens
// the phone's own AR, the real dimensions, and a QR code so the person sitting at a
// desktop can be holding it in ten seconds. No build step, no framework, no tracking.
//
// The one thing a static host gets wrong on its own is the USDZ content type: served
// as application/octet-stream, iOS opens nothing and says nothing. vercel.json here
// sets it, and the same three lines are in the README for any other host.
//
// Usage:
//   node tools/dropin.mjs <projectId> --out <dir> [--name "Boucle Lounge Chair"]
//        [--brand "Workflo"] [--note "…"] [--url https://client.example.com] [--deploy]
import { mkdir, writeFile, readFile, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import sharp from 'sharp';

const run = promisify(execFile);
const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

const argv = process.argv.slice(2);
const id = argv[0];
const flag = (name, fallback = null) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : (argv.includes(`--${name}`) ? true : fallback);
};
if (!id || id.startsWith('--')) {
  console.error('usage: node tools/dropin.mjs <projectId> --out <dir> [--name …] [--brand …] [--note …] [--url …] [--deploy]');
  process.exit(1);
}
const dir = join(ROOT, 'projects', id);
const out = resolve(String(flag('out', join(ROOT, 'dropin', id))));
const meta = JSON.parse(await readFile(join(dir, 'blank.json'), 'utf8'));
if (!existsSync(join(dir, 'model.glb'))) {
  console.error(`${id} has not been baked: open it in the studio and press "Build the AR model" first.`);
  process.exit(1);
}
const name = String(flag('name', 'The product'));
const brand = String(flag('brand', ''));
const note = String(flag('note', ''));
let url = flag('url', null);

const cm = (m) => (m * 100).toFixed(0);
const dims = meta.dimensions;
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

await mkdir(join(out, 'vendor'), { recursive: true });

// files: the compressed twin for the viewer, the plain one as a fallback and download,
// the USDZ for iOS, and a poster so the page shows something before the model lands
const lite = existsSync(join(dir, 'model-lite.glb')) ? 'model-lite.glb' : 'model.glb';
for (const f of ['model.glb', 'model.usdz', lite]) {
  if (existsSync(join(dir, f))) await copyFile(join(dir, f), join(out, f));
}
await copyFile(join(ROOT, 'node_modules', '@google', 'model-viewer', 'dist', 'model-viewer.min.js'),
  join(out, 'vendor', 'model-viewer.min.js'));
// the Draco decoder, so the compressed twin works on a host with no internet assumptions
await mkdir(join(out, 'vendor', 'draco'), { recursive: true });
for (const f of ['draco_decoder.js', 'draco_decoder.wasm', 'draco_wasm_wrapper.js']) {
  const src = join(ROOT, 'node_modules', 'three', 'examples', 'jsm', 'libs', 'draco', 'gltf', f);
  if (existsSync(src)) await copyFile(src, join(out, 'vendor', 'draco', f));
}
// poster: the project's own source photograph, dimmed, so the first paint is the product
let poster = null;
for (const cand of ['source-front.png', 'source.png']) {
  if (!existsSync(join(dir, cand))) continue;
  await sharp(join(dir, cand)).resize(1200, 1200, { fit: 'inside' }).jpeg({ quality: 78 }).toFile(join(out, 'poster.jpg'));
  poster = 'poster.jpg';
  break;
}

// QR, when the address is known. Client-side generation would need a QR encoder in the
// page; the person running this is about to deploy and knows the name they chose.
let qr = '';
if (typeof url === 'string') {
  try {
    const QR = (await import('qrcode')).default;
    // currentColor, not a baked hex: the page has a light and a dark theme, and a QR
    // in the light theme's ink is invisible on the dark one
    const svg = (await QR.toString(url, { type: 'svg', margin: 1, color: { dark: '#000000', light: '#0000' } }))
      .replace(/#000000/g, 'currentColor').replace(/#000/g, 'currentColor');
    qr = `<figure class="qr"><div class="qr-svg">${svg}</div><figcaption>Point your phone's camera here<br><span>${escapeHtml(url)}</span></figcaption></figure>`;
  } catch { /* no qr module: the page simply has no code on it */ }
}

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%231f6feb'/%3E%3Cpath d='M4 10.5 8 4l4 6.5z' fill='%23fff'/%3E%3C/svg%3E">
<title>${escapeHtml(name)}${brand ? ` · ${escapeHtml(brand)}` : ''}</title>
<meta name="description" content="${escapeHtml(name)} in 3D, at true size, ready to stand in your own room.">
<meta property="og:title" content="${escapeHtml(name)}">
<meta property="og:description" content="See it in your room, at ${cm(dims.width)} by ${cm(dims.depth)} by ${cm(dims.height)} cm.">
${poster ? `<meta property="og:image" content="${poster}">` : ''}
<style>
  :root {
    --ink: #111318; --dim: #6b7180; --line: #e3e1dc; --paper: #f6f5f2; --card: #ffffff;
    --accent: #1f6feb;
    --sans: ui-sans-serif, -apple-system, "Segoe UI", Roboto, system-ui, sans-serif;
    --mono: ui-monospace, "SF Mono", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ink: #eceae6; --dim: #9aa0ad; --line: #262a33; --paper: #0d0f14; --card: #14171e;
      --accent: #6ea8ff;
    }
  }
  :root[data-theme="dark"] {
    --ink: #eceae6; --dim: #9aa0ad; --line: #262a33; --paper: #0d0f14; --card: #14171e;
    --accent: #6ea8ff;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body { background: var(--paper); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 28px 20px 56px; }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 22px; }
  .brand { font-family: var(--mono); font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--dim); }
  h1 { font-size: clamp(28px, 4.4vw, 44px); line-height: 1.1; margin: 0; letter-spacing: -0.02em; text-wrap: balance; }
  .stage { position: relative; background: var(--card); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
  model-viewer { width: 100%; height: min(66vh, 620px); background: transparent; --poster-color: transparent; }
  .bar { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;
    padding: 14px 16px; border-top: 1px solid var(--line); }
  .size { font-family: var(--mono); font-size: 13px; color: var(--dim); font-variant-numeric: tabular-nums; }
  .size b { color: var(--ink); font-weight: 500; }
  .cta { appearance: none; border: 0; border-radius: 999px; background: var(--accent); color: #fff;
    font: 600 15px/1 var(--sans); padding: 13px 22px; cursor: pointer; }
  .cta:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }
  .hint { font-size: 13px; color: var(--dim); }
  .below { display: grid; grid-template-columns: 1fr; gap: 22px; margin-top: 26px; }
  @media (min-width: 780px) { .below.has-qr { grid-template-columns: 1fr auto; } }
  .note { max-width: 62ch; margin: 0; color: var(--dim); }
  .note strong { color: var(--ink); font-weight: 600; }
  .qr { display: flex; gap: 14px; align-items: center; margin: 0; padding: 14px;
    background: var(--card); border: 1px solid var(--line); border-radius: 12px; }
  .qr-svg { width: 104px; height: 104px; flex: none; color: var(--ink); }
  .qr-svg svg { width: 100%; height: 100%; display: block; }
  .qr figcaption { font-size: 13px; color: var(--dim); }
  .qr figcaption span { font-family: var(--mono); font-size: 11px; word-break: break-all; }
  @media (max-width: 779px) { .qr { display: none; } }
  footer { margin-top: 34px; padding-top: 16px; border-top: 1px solid var(--line);
    font-size: 12.5px; color: var(--dim); display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  footer a { color: inherit; }
  @media (prefers-reduced-motion: reduce) { model-viewer { --auto-rotate: none; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      ${brand ? `<div class="brand">${escapeHtml(brand)}</div>` : ''}
      <h1>${escapeHtml(name)}</h1>
    </div>
    <div class="hint">Drag to turn it. On a phone, stand it on your floor.</div>
  </header>

  <div class="stage">
    <model-viewer id="mv"
      src="${lite}"
      ios-src="model.usdz"
      ${poster ? `poster="${poster}"` : ''}
      alt="${escapeHtml(name)} in three dimensions"
      camera-controls auto-rotate rotation-per-second="14deg"
      touch-action="pan-y" interaction-prompt="auto"
      ar ar-modes="webxr scene-viewer quick-look" ar-scale="fixed"
      shadow-intensity="0.85" shadow-softness="0.9" exposure="1"
      environment-image="neutral" tone-mapping="commerce"
      min-camera-orbit="auto auto 40%" max-camera-orbit="auto auto 220%">
      <button slot="ar-button" type="button" class="cta">View in your room</button>
    </model-viewer>
    <div class="bar">
      <div class="size">Actual size <b>${cm(dims.width)} &times; ${cm(dims.depth)} &times; ${cm(dims.height)} cm</b> (w &times; d &times; h)</div>
      <div class="hint" id="ar-hint">Open this page on a phone to place it in your room.</div>
    </div>
  </div>

  <div class="below${qr ? ' has-qr' : ''}">
    <p class="note">${note ? escapeHtml(note) : `Shown at its real size. <strong>Pinch-to-resize is switched off on purpose</strong>, because the question worth answering is whether it fits.`}</p>
    ${qr}
  </div>

  <footer>
    <span>${brand ? escapeHtml(brand) : 'Made with Photo to AR'}</span>
    <span><a href="model.glb" download>glTF</a> &middot; <a href="model.usdz" download>USDZ</a></span>
  </footer>
</div>

<script type="module" src="vendor/model-viewer.min.js"></script>
<script type="module">
  const mv = document.querySelector('#mv');
  customElements.whenDefined('model-viewer').then(() => {
    const MV = customElements.get('model-viewer');
    // the decoder ships beside the page, so this works on a host with no CDN reach
    MV.dracoDecoderLocation = new URL('vendor/draco/', location.href).href;
  });
  // the compressed twin is the fast path; the plain file is the one that always works
  let fellBack = false;
  mv.addEventListener('error', () => {
    if (fellBack) return;
    fellBack = true;
    mv.src = 'model.glb';
  });
  // say what will actually happen when the button is pressed
  mv.addEventListener('load', () => {
    const hint = document.getElementById('ar-hint');
    if (mv.canActivateAR) hint.textContent = 'Tap "View in your room" and point at the floor.';
  });
</script>
</body>
</html>
`;

await writeFile(join(out, 'index.html'), html);

// Static hosting needs to be told about USDZ, and the model files never change.
await writeFile(join(out, 'vercel.json'), JSON.stringify({
  $schema: 'https://openapi.vercel.sh/vercel.json',
  cleanUrls: true,
  headers: [
    { source: '/(.*).usdz', headers: [{ key: 'Content-Type', value: 'model/vnd.usdz+zip' }] },
    { source: '/(.*).glb', headers: [{ key: 'Content-Type', value: 'model/gltf-binary' }] },
    { source: '/vendor/(.*)', headers: [{ key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }] },
  ],
}, null, 2) + '\n');

await writeFile(join(out, 'README.txt'), `${name}${brand ? ` (${brand})` : ''}
Built from photo-to-ar project ${id} on ${new Date().toISOString().slice(0, 10)}.

Deploy:
  cd ${out}
  npx vercel --prod

Any other static host works too, as long as it serves .usdz as
"model/vnd.usdz+zip". Served as application/octet-stream, the iOS AR button
does nothing and reports no error, which is a miserable thing to debug.

To put the QR code on the page, rebuild with the address it will live at:
  node tools/dropin.mjs ${id} --out ${out} --url https://your-address
`);

const bytes = async (f) => (existsSync(join(out, f)) ? (await readFile(join(out, f))).length : 0);
const total = (await Promise.all(['index.html', lite, 'model.glb', 'model.usdz', 'poster.jpg', 'vendor/model-viewer.min.js'].map(bytes)))
  .reduce((a, b) => a + b, 0);
console.log(`${out}`);
console.log(`  ${name}${brand ? ` (${brand})` : ''}, ${cm(dims.width)} x ${cm(dims.depth)} x ${cm(dims.height)} cm`);
console.log(`  viewer model ${((await bytes(lite)) / 1e6).toFixed(2)} MB, usdz ${((await bytes('model.usdz')) / 1e6).toFixed(2)} MB, page total ${(total / 1e6).toFixed(2)} MB${qr ? ', QR included' : ''}`);

if (flag('deploy')) {
  console.log('\ndeploying…');
  const { stdout } = await run('npx', ['--yes', 'vercel', '--prod', '--yes'], { cwd: out, maxBuffer: 1 << 24 });
  const live = (stdout.match(/https:\/\/[^\s]+\.vercel\.app/g) || []).pop();
  console.log(live ? `live at ${live}` : stdout.trim().split('\n').slice(-3).join('\n'));
  if (live && typeof url !== 'string') {
    console.log('rerun with --url ' + live + ' to put the QR code on the page');
  }
}
