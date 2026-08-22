/**
 * Minimal ComfyUI client: submit a workflow, wait, pull the outputs.
 *
 * Vendored from ~/product-viz/tools/comfy.mjs on 2026-08-20. That copy and this one
 * are now independent; if you fix a bug in one, the other will not hear about it.
 */
const COMFY = process.env.COMFY_URL || 'http://127.0.0.1:8188';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** Is the server there at all? Cheap enough to call before every job. */
export async function alive() {
  // Three tries over about twenty seconds. ComfyUI's loop stalls for several seconds
  // while it unloads one model and loads another, and a single four-second probe in
  // that window reported a perfectly healthy server as dead, which aborted a stage.
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(`${COMFY}/system_stats`, { signal: AbortSignal.timeout(6000) });
      if (r.ok) return true;
    } catch { /* try again */ }
    await new Promise((res) => setTimeout(res, 4000));
  }
  return false;
}

/** Submit a prompt graph and resolve once it finishes. Returns the history entry. */
export async function run(workflow, { label = 'job', timeoutMs = 900_000, onProgress } = {}) {
  const post = await fetch(`${COMFY}/prompt`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt: workflow, client_id: 'photo-to-ar' }),
  });
  if (!post.ok) throw new Error(`ComfyUI rejected ${label}: ${post.status} ${await post.text()}`);
  const { prompt_id: id } = await post.json();

  const deadline = Date.now() + timeoutMs;
  let lastNote = 0;
  while (Date.now() < deadline) {
    await wait(1500);
    const history = await (await fetch(`${COMFY}/history/${id}`)).json();
    const entry = history[id];
    if (entry) {
      if (entry.status?.status_str === 'error') {
        const msg = entry.status?.messages?.find((m) => m[0] === 'execution_error')?.[1];
        throw new Error(`${label} failed: ${msg?.exception_message || 'unknown'}\n`
          + (msg?.traceback || []).slice(-6).join(''));
      }
      return entry;
    }
    if (Date.now() - lastNote > 5_000) {
      lastNote = Date.now();
      const q = await (await fetch(`${COMFY}/queue`)).json();
      onProgress?.({ running: q.queue_running?.length || 0, pending: q.queue_pending?.length || 0 });
    }
  }
  throw new Error(`${label} timed out after ${Math.round(timeoutMs / 1000)}s`);
}

/** Every output file the run produced, as {filename, subfolder, type, kind}. */
export function outputsOf(entry) {
  const files = [];
  for (const out of Object.values(entry.outputs || {})) {
    for (const kind of ['images', 'gifs', '3d', 'result', 'files', 'meshes']) {
      for (const f of out[kind] || []) {
        if (f && typeof f === 'object' && f.filename) files.push({ ...f, kind });
      }
    }
  }
  return files;
}

/** Download one output file. */
export async function fetchOutput(file) {
  const qs = new URLSearchParams({
    filename: file.filename,
    subfolder: file.subfolder || '',
    type: file.type || 'output',
  });
  const r = await fetch(`${COMFY}/view?${qs}`);
  if (!r.ok) throw new Error(`could not fetch ${file.filename}: ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

/** Upload a local image so LoadImage can see it. */
export async function upload(buffer, name) {
  const fd = new FormData();
  fd.append('image', new Blob([buffer], { type: 'image/png' }), name);
  fd.append('overwrite', 'true');
  const r = await fetch(`${COMFY}/upload/image`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`upload refused: ${r.status}`);
  const j = await r.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

/** Which nodes and models this ComfyUI actually has. */
export async function objectInfo() {
  return (await fetch(`${COMFY}/object_info`)).json();
}

export { COMFY };
