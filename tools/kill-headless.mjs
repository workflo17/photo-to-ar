// Stop leftover headless Edge instances started by tools/shot.mjs and tools/test.mjs.
// Matches on the dedicated profile directory (edge-p2ar), never on the process name,
// so the person's own Edge windows are untouched.
import { execFileSync } from 'node:child_process';
const out = execFileSync('wmic', ['process', 'where', "commandline like '%edge-p2ar%'", 'get', 'processid'], { encoding: 'utf8' });
const pids = out.split(/\r?\n/).map((l) => l.trim()).filter((l) => /^\d+$/.test(l));
for (const pid of pids) { try { process.kill(Number(pid), 'SIGKILL'); } catch {} }
console.log(`stopped ${pids.length} headless edge process${pids.length === 1 ? '' : 'es'}`);
