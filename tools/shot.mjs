// Screenshot the studio in a browser that really composites.
//
// The embedded preview pane reports visibilityState "hidden" when it is not on
// screen, which gives every canvas a zero-sized layout box and never fires an
// IntersectionObserver. WebGL there renders nothing and proves nothing. Headless
// Edge paints for real.
//
// Usage: node tools/shot.mjs <url> <out.png> [clickSelector] [waitMs]
import { spawn } from 'node:child_process';
import { writeFileSync, existsSync } from 'node:fs';

const [url, out, clickSel, waitMs = '4500'] = process.argv.slice(2);
const EDGE = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const PORT = 9240;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const proc = spawn(EDGE, [
  '--headless=new', `--remote-debugging-port=${PORT}`, '--remote-allow-origins=*',
  `--user-data-dir=${process.env.TEMP || '.'}/edge-p2ar`, '--no-first-run',
  '--window-size=1600,1000', '--hide-scrollbars',
  // headless defaults to a software rasteriser that refuses WebGL2 outright
  '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--enable-webgl',
  'about:blank',
], { stdio: 'ignore' });

let targets = null;
for (let i = 0; i < 60 && !targets; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); } catch { await sleep(400); }
}
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let id = 0; const pending = new Map(); const errors = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errors.push(m.params.entry.text);
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
};
const send = (method, params = {}) => new Promise((res) => {
  const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (e) => (await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true })).result?.result?.value;

await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable');
await send('Page.navigate', { url });
await sleep(3000);

if (clickSel) {
  console.log('click:', await evaluate(`(()=>{const el=document.querySelector('${clickSel}'); if(!el) return 'NOT FOUND'; el.click(); return 'clicked';})()`));
  await sleep(Number(waitMs));
}

console.log('state:', await evaluate(`JSON.stringify({
  regions: document.querySelectorAll('.region').length,
  canvas: [document.querySelector('#view')?.width, document.querySelector('#view')?.height],
  webgl2: !!document.createElement('canvas').getContext('webgl2'),
  studio: document.querySelector('#stage-studio') ? !document.querySelector('#stage-studio').hidden : null
})`));

const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) { writeFileSync(out, Buffer.from(shot.result.data, 'base64')); console.log('wrote', out); }
else console.error('capture failed');
console.log('errors:', errors.length ? errors.slice(0, 4) : 'none');
proc.kill();
process.exit(0);
