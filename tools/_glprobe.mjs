import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
const EDGE = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const PORT = 9335; const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const proc = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${PORT}`, '--remote-allow-origins=*', `--user-data-dir=${process.env.TEMP}/edge-p2ar-probe`, '--no-first-run', '--window-size=1600,900', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--enable-webgl'], { stdio: 'ignore' });
let targets = null; for (let i = 0; i < 60 && !targets; i++) { try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); } catch { await sleep(400); } }
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let id = 0; const pending = new Map();
ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (e) => (await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true })).result?.result?.value;
await send('Page.enable'); await send('Runtime.enable');
await send('Page.navigate', { url: 'about:blank' }); await sleep(1000);
console.log(await evaluate(`(()=>{const c=document.createElement('canvas');const gl=c.getContext('webgl2');const x=gl.getExtension('WEBGL_debug_renderer_info');return x?gl.getParameter(x.UNMASKED_RENDERER_WEBGL):'no ext: '+gl.getParameter(gl.RENDERER);})()`));
proc.kill(); process.exit(0);
