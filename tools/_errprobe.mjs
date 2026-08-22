import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
const EDGE = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const PORT = 9334;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const proc = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${PORT}`, '--remote-allow-origins=*', `--user-data-dir=${process.env.TEMP}/edge-p2ar-probe`, '--no-first-run', '--window-size=1600,900', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--enable-webgl'], { stdio: 'ignore' });
let targets = null;
for (let i = 0; i < 60 && !targets; i++) { try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); } catch { await sleep(400); } }
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let id = 0; const pending = new Map(); const logs = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled') logs.push(m.params.type + ': ' + m.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 300));
  if (m.method === 'Runtime.exceptionThrown') logs.push('EXC: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 400));
  if (m.method === 'Log.entryAdded') logs.push('LOG ' + m.params.entry.level + ': ' + m.params.entry.text.slice(0, 300));
};
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (e) => (await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true })).result?.result?.value;
await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable');
await send('Page.navigate', { url: process.argv[2] || 'http://localhost:5140' });
await sleep(3000);
if (process.argv[3]) await evaluate(`document.querySelector('${process.argv[3]}').click()`);
await sleep(15000);
console.log(logs.join('\n') || 'no console output');
if (process.env.EVAL) console.log('eval:', await evaluate(process.env.EVAL));
const t0 = Date.now();
const shot = await Promise.race([send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: process.env.FULLPAGE === '1' }), sleep(Number(process.argv[4] || 40000)).then(() => null)]);
if (shot?.result?.data) (await import('node:fs')).writeFileSync(process.argv[5] || 'probe.png', Buffer.from(shot.result.data, 'base64'));
console.log('capture', shot ? 'ok' : 'TIMED OUT', ((Date.now() - t0) / 1000).toFixed(1) + 's');
proc.kill(); process.exit(0);
