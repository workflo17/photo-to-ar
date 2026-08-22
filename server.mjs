/**
 * The app server.
 *
 * Deliberately shallow: it parses requests, calls one deep module, and streams the
 * result. Anything that looks like domain logic in here belongs in lib/ instead.
 *
 * It binds to every interface rather than to localhost, because the point of the
 * whole exercise is putting a model on a phone, and a phone cannot reach 127.0.0.1.
 */
import { createServer } from 'node:http';
import { createServer as createHttps } from 'node:https';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { readFile, writeFile, mkdir, readdir, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createGzip } from 'node:zlib';
import { join, extname, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { networkInterfaces } from 'node:os';

import { readiness, reconstruct, reconstructViews } from './lib/recon.mjs';
import { readiness as colourReadiness, reconstructColour } from './lib/colour.mjs';
import { makeBlank } from './lib/blank.mjs';
import { bake } from './lib/bake.mjs';
import { catalogue, createMaterial, deleteMaterial, loadCustom, WEAVES } from './lib/materials.mjs';
import { emptyLook, validate } from './lib/look.mjs';

const runPy = promisify(execFile);

/**
 * Score how well the finished mesh matches the photograph it came from.
 *
 * Silhouette IoU between the mesh, rendered from the best-fitting camera, and the
 * matte's own subject mask. Around 0.9 would mean the reconstruction is the same shape
 * as the thing photographed; 0.7 means it is a plausible cousin of it.
 */
/**
 * The two stages that need the photographs after the Blank exists: the silhouette
 * score, and painting the photographs onto the colour atlas. Both are best-effort:
 * a failure is reported and the Blank stands as it is.
 */
async function finishColour(dir, meta, ch, blankOpts = null, pass = 1) {
  let known = {};
  try {
    const fit = await scoreFit(dir);
    Object.assign(meta, { fitScore: fit.fitScore, fitView: fit.fitView, fit: fit.fit });
    known = { front: fit.best };
    ch.send('progress', { stage: 'fit', detail: `silhouette match ${(fit.fitScore * 100).toFixed(0)}% at ${fit.fitView}` });
  } catch (e) {
    ch.send('progress', { stage: 'fit', detail: `could not score the fit: ${e.message}` });
  }
  if (meta.hasColour) {
    try {
      const { projectPhotos } = await import('./lib/project.mjs');
      const { subjectMask } = await import('./lib/colour.mjs');
      const used = await projectPhotos(dir, {
        maskFor: subjectMask,
        known,
        log: (m) => ch.send('progress', { stage: 'project', detail: m }),
      });
      meta.projected = used;
      meta.refused = projectPhotos.lastRefused || [];
      if (used.length) ch.send('progress', { stage: 'project', detail: `${used.length} photograph${used.length === 1 ? '' : 's'} painted onto the model` });
      if (meta.refused.length) ch.send('progress', { stage: 'project', detail: `${meta.refused.length} photograph${meta.refused.length === 1 ? '' : 's'} refused: ${meta.refused.map((r) => `${r.side} (${r.reason})`).join('; ')}` });

      // Second pass: when two or more photographs fit well, the reconstruction's
      // invented volume can be carved back to their silhouettes, which the first
      // pass could not do because the alignment needs a Blank to align against.
      // One extra makeBlank and one extra projection, once; not recursive.
      // 85%, not 70%: measured on the figure, a three-quarter view fitted at 79%
      // carved real hip and leg volume and took the front's own fit from 87% to
      // 78%. A silhouette may only carve when it is placed almost exactly.
      const good = (projectPhotos.lastSilhouettes || []).filter((sil) => sil.score >= 0.85);
      if (pass === 1 && blankOpts && meta.frame && good.length >= 2 && blankOpts.carve) {
        ch.send('progress', { stage: 'finish', detail: `second pass: carving with the silhouettes of ${good.length} photographs` });
        const { engine } = meta;
        const again = await makeBlank(join(dir, 'raw.glb'), dir, {
          ...blankOpts,
          carve: { silhouettes: good, frame: meta.frame },
        });
        again.engine = engine;
        await writeFile(join(dir, 'blank.json'), JSON.stringify(again, null, 2));
        await writeFile(join(dir, 'look.json'), JSON.stringify(emptyLook(again.regions)));
        return finishColour(dir, again, ch, blankOpts, 2);
      }
    } catch (e) {
      ch.send('progress', { stage: 'project', detail: `could not project the photographs: ${e.message}` });
    }
  }
  await writeFile(join(dir, 'blank.json'), JSON.stringify(meta, null, 2));
  return meta;
}

async function scoreFit(dir) {
  const { NodeIO } = await import('@gltf-transform/core');
  const { findView } = await import('./lib/align.mjs');
  // The flood-fill matte cannot separate a figure from a studio gradient: on the
  // Shin Godzilla photographs it kept 77 to 89% of the frame, the "silhouette" was a
  // rectangle, and the fit scored 82% against a blob. Use the segmentation model
  // for the alignment mask whenever it is available; the flood fill is the fallback.
  try {
    const { subjectMask } = await import('./lib/colour.mjs');
    const png = await subjectMask(await readFile(join(dir, 'source.png')));
    await writeFile(join(dir, 'matted-alpha.png'), png);
  } catch { /* no segmentation model: photomask.py falls back to the matte's alpha */ }
  await runPy('python', [join(ROOT, 'lib', 'photomask.py'),
    join(dir, 'matted.png'), join(dir, 'mask.bin'), '192']);
  const doc = await new NodeIO().read(join(dir, 'blank.glb'));
  const prim = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
  const photo = new Uint8Array(await readFile(join(dir, 'mask.bin')));
  // 10-degree steps, the same search the projection uses: at 15 the Shin Godzilla
  // front scored 70% while the projection found 87% for the same photograph
  const { best } = findView(prim.getAttribute('POSITION').getArray(),
    prim.getIndices().getArray(), photo, 192, 10);
  return { fitScore: best.score, fitView: best.view.name, fit: best.fit, best };
}
const ROOT = dirname(fileURLToPath(import.meta.url));
let lan = [];   // this machine's LAN addresses, filled in at startup

/* Quality presets, measured on a 3080 against the reference chair photograph:
 *
 *   draft     surface net 256   ~85s    603k raw triangles
 *   standard  surface net 384   (between the two)
 *   fine      surface net 512   ~220s   2.0M raw triangles
 *
 * Plain marching cubes ("basic") is not offered. It is roughly twice as fast as
 * surface net at the same grid and it looks it: the whole surface arrives covered
 * in stair-step ripples that no amount of downstream simplification removes.
 */
const QUALITY = {
  draft: { algorithm: 'surface net', octreeResolution: 256, steps: 30 },
  standard: { algorithm: 'surface net', octreeResolution: 384, steps: 50 },
  fine: { algorithm: 'surface net', octreeResolution: 512, steps: 50 },
};
const PROJECTS = join(ROOT, 'projects');
const PORT = Number(process.env.PORT || 5140);

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.glb': 'model/gltf-binary',
  // Safari decides whether to hand a file to AR Quick Look by its content type. A
  // host that guesses application/octet-stream leaves the AR button doing nothing,
  // with no error anywhere, which is a miserable thing to debug on a phone.
  '.usdz': 'model/vnd.usdz+zip',
};

const send = (res, code, body, type = 'application/json; charset=utf-8', extra = {}) => {
  res.writeHead(code, { 'content-type': type, 'cache-control': 'no-store', ...extra });
  res.end(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body));
};

/* Types that gzip meaningfully. JPEG and PNG are already entropy-coded; USDZ looks
 * compressed because it is a zip, but the spec requires its entries be STORED, not
 * deflated, so the whole package gzips to roughly half. glTF binary is mostly float
 * buffers and drops by a third. The model-viewer bundle, the single biggest download
 * on the AR page, drops from about a megabyte to a quarter of one. */
const COMPRESSIBLE = new Set(['.js', '.mjs', '.css', '.html', '.json', '.svg', '.glb', '.usdz', '.bin']);

async function serveFile(req, res, path, { download, longCache } = {}) {
  try {
    const s = await stat(path);
    if (!s.isFile()) throw new Error('not a file');
    const ext = extname(path).toLowerCase();
    const type = MIME[ext] || 'application/octet-stream';

    // Revalidation instead of no-store. Model files are rewritten under the same name
    // during a design session, so they can never be trusted blindly, but no-store made
    // every AR launch a full re-download even when nothing had changed. An ETag from
    // mtime and size gives the phone a 304 when the file is the same and fresh bytes
    // the moment a rebake touches it.
    const etag = `"${Math.round(s.mtimeMs)}-${s.size}"`;
    if (req.headers['if-none-match'] === etag) {
      res.writeHead(304, { etag, 'cache-control': longCache ? 'public, max-age=86400' : 'no-cache' });
      return res.end();
    }

    const gzip = COMPRESSIBLE.has(ext) && s.size > 1024
      && (req.headers['accept-encoding'] || '').includes('gzip');

    res.writeHead(200, {
      'content-type': type,
      etag,
      vary: 'accept-encoding',
      'cache-control': longCache ? 'public, max-age=86400' : 'no-cache',
      'access-control-allow-origin': '*',
      ...(gzip ? { 'content-encoding': 'gzip' } : { 'content-length': s.size }),
      ...(download ? { 'content-disposition': `attachment; filename="${download}"` } : {}),
    });
    const stream = createReadStream(path);
    if (gzip) stream.pipe(createGzip({ level: 6 })).pipe(res);
    else stream.pipe(res);
  } catch {
    send(res, 404, { error: 'not found' });
  }
}

/** A Server-Sent Events channel, which is what a multi-minute GPU job wants. */
function sse(res) {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-store',
    connection: 'keep-alive',
    'x-accel-buffering': 'no',
  });
  return {
    send: (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
    end: () => res.end(),
  };
}

async function readBody(req, limit = 64 * 1024 * 1024) {
  const chunks = [];
  let total = 0;
  for await (const c of req) {
    total += c.length;
    if (total > limit) throw new Error('body too large');
    chunks.push(c);
  }
  return Buffer.concat(chunks);
}

const projectDir = (id) => {
  // an id from the URL must never escape the projects directory
  if (!/^[a-z0-9-]{6,40}$/.test(id)) return null;
  const dir = join(PROJECTS, id);
  return resolve(dir).startsWith(resolve(PROJECTS)) ? dir : null;
};

/* --------------------------------------------------------------------- routes */
const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const path = url.pathname;

  try {
    if (path === '/api/status') {
      const [r, c] = await Promise.all([readiness(), colourReadiness()]);
      return send(res, 200, {
        gpu: r.ok, reason: r.reason || null,
        colour: c.ok, colourReason: c.reason || null,
        materials: catalogue(),
      });
    }

    if (path === '/api/projects' && req.method === 'GET') {
      const ids = await readdir(PROJECTS).catch(() => []);
      const list = [];
      for (const id of ids) {
        if (id.startsWith('_')) continue;
        try {
          const meta = JSON.parse(await readFile(join(PROJECTS, id, 'blank.json'), 'utf8'));
          list.push({ id, ...meta });
        } catch { /* half-built project, skip it */ }
      }
      list.sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''));
      return send(res, 200, list);
    }

    // Upload a photograph and build a Blank from it. Streams progress, because the
    // reconstruction is the slowest thing here by two orders of magnitude.
    if (path === '/api/build' && req.method === 'POST') {
      const heightM = Number(url.searchParams.get('height') || 0.78);
      const widthM = Number(url.searchParams.get('width')) || null;
      const depthM = Number(url.searchParams.get('depth')) || null;
      const seed = Number(url.searchParams.get('seed') || 7);
      const image = await readBody(req);
      if (image.length < 1024) return send(res, 400, { error: 'no image received' });

      const quality = QUALITY[url.searchParams.get('quality')] || QUALITY.standard;
      const id = randomUUID().slice(0, 12);
      const dir = join(PROJECTS, id);
      await mkdir(dir, { recursive: true });

      // Either a bare image body (one view) or JSON carrying up to four. Base64 costs
      // a third more bytes than multipart would, and buys not having to hand-write a
      // multipart parser for a server that only ever talks to its own page.
      // Views carry roles. The four cardinal sides feed the shape engine; every view
      // feeds the colour projection unless it is listed in `shapeOnly` (a photograph
      // of the same mould in another colourway: right shape, wrong colour). A view
      // under any other key (a close-up, say) is colour-only by construction, since
      // the shape engine only takes the cardinal four.
      let views, shapeOnly = [];
      if (image[0] === 0x7b) {
        const body = JSON.parse(image.toString('utf8'));
        views = Object.fromEntries(Object.entries(body.views || {})
          .filter(([k, v]) => /^[a-z0-9-]{1,24}$/.test(k) && typeof v === 'string' && v.length > 100)
          .map(([k, v]) => [k, Buffer.from(v.split(',').pop(), 'base64')]));
        shapeOnly = Array.isArray(body.shapeOnly) ? body.shapeOnly.filter((k) => views[k]) : [];
      } else {
        views = { front: image };
      }
      if (!views.front) return send(res, 400, { error: 'a front view is required' });
      for (const [side, buf] of Object.entries(views)) {
        await writeFile(join(dir, `source-${side}.png`), buf);
      }
      await writeFile(join(dir, 'source.png'), views.front);
      const CARDINAL = ['front', 'left', 'back', 'right'];
      const roles = Object.fromEntries(Object.keys(views).map((k) => [k, { shape: CARDINAL.includes(k), colour: !shapeOnly.includes(k) }]));
      await writeFile(join(dir, 'views.json'), JSON.stringify({ sides: roles }, null, 2));
      const shapeViews = Object.fromEntries(Object.entries(views).filter(([k]) => roles[k].shape));

      const ch = sse(res);
      ch.send('start', { id });
      try {
        // Cut the backdrop away BEFORE the model sees the photo. Removing geometry
        // that was never generated beats cutting a reconstructed floor slab off
        // afterwards, and it stops the slab distorting the scale measurement.
        // If no backdrop is found the script exits 2 and the original is used.
        const fed = {};
        for (const [side, buf] of Object.entries(views)) {
          ch.send('progress', { stage: 'matte', detail: `${side}: looking for a backdrop` });
          try {
            const { stdout } = await runPy('python', [join(ROOT, 'lib', 'matte.py'),
              join(dir, `source-${side}.png`), join(dir, `matted-${side}.png`)]);
            fed[side] = await readFile(join(dir, `matted-${side}.png`));
            ch.send('progress', { stage: 'matte', detail: `${side}: ${stdout.trim()}` });
          } catch {
            fed[side] = buf;
            ch.send('progress', { stage: 'matte', detail: `${side}: no backdrop found, using it as it is` });
          }
        }
        // The flood-fill matte cannot separate a product from a studio gradient, and
        // what it leaves in (a caption strip, a patch of backdrop) the shape engine
        // reconstructs faithfully: a figure came back with a bar through its head.
        // When the segmentation model is up, every view is matted with it instead:
        // subject kept, everything else white, which is what the shape engine was
        // trained on. The flood fill stays as the fallback.
        try {
          const { subjectMask } = await import('./lib/colour.mjs');
          const sharp = (await import('sharp')).default;
          for (const side of Object.keys(views)) {
            const maskPng = await subjectMask(views[side]);
            const mask = await sharp(maskPng).toColourspace('b-w').raw().toBuffer({ resolveWithObject: true });
            const src = await sharp(views[side]).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
            if (mask.info.width !== src.info.width || mask.info.height !== src.info.height) continue;
            const out = Buffer.alloc(src.info.width * src.info.height * 3);
            const alpha = Buffer.alloc(src.info.width * src.info.height);
            for (let i = 0; i < alpha.length; i++) {
              const a = mask.data[i * mask.info.channels] / 255;
              alpha[i] = Math.round(a * 255);
              for (let k = 0; k < 3; k++) out[i * 3 + k] = Math.round(src.data[i * 4 + k] * a + 255 * (1 - a));
            }
            const png = await sharp(out, { raw: { width: src.info.width, height: src.info.height, channels: 3 } }).png().toBuffer();
            await writeFile(join(dir, `matted-${side}.png`), png);
            await sharp(alpha, { raw: { width: src.info.width, height: src.info.height, channels: 1 } }).png().toFile(join(dir, `matted-${side}-alpha.png`));
            fed[side] = png;
            ch.send('progress', { stage: 'matte', detail: `${side}: segmented, ${(100 * alpha.reduce((t, v) => t + (v > 127), 0) / alpha.length).toFixed(0)}% of the frame is the subject` });
          }
        } catch (e) {
          ch.send('progress', { stage: 'matte', detail: `segmentation matte unavailable (${e.message}), keeping the flood fill` });
        }
        if (fed.front) await writeFile(join(dir, 'matted.png'), fed.front);

        // Two engines. Hunyuan3D returns shape only and is what the synthetic fabric
        // library is for; TripoSplat returns the product's own colour, which is what a
        // buyer actually wants to see, at the cost of a slower run and a mesh the
        // finish pass has to work harder on.
        const wanted = url.searchParams.get('engine');
        const engine = wanted === 'colour' ? 'colour' : wanted === 'hybrid' ? 'hybrid' : 'shape';
        if (engine === 'hybrid') {
          // Shape from every photograph through Hunyuan3D, colour from the front one
          // through TripoSplat, registered and merged (lib/hybrid.mjs). The colour
          // engine alone invents the sides it cannot see; on a figure that turned the
          // dorsal plates into a row of knobs. The shape engine with all views gets
          // the plates and returns no colour. Together they get both.
          ch.send('progress', { stage: 'recon', detail: 'shape from all photographs' });
          await reconstructViews(Object.fromEntries(Object.entries(fed).filter(([k]) => shapeViews[k])), join(dir, 'raw-shape.glb'), {
            seed, ...quality,
            onProgress: (p) => ch.send('progress', { stage: 'recon', detail: p.detail }),
          });
          ch.send('progress', { stage: 'recon', detail: 'colour from the front photograph' });
          await reconstructColour(views.front, join(dir, 'raw-colour.glb'), {
            seed,
            onProgress: (p) => ch.send('progress', { stage: 'recon', detail: p.detail }),
          });
          const { hybridRaw } = await import('./lib/hybrid.mjs');
          await hybridRaw(join(dir, 'raw-shape.glb'), join(dir, 'raw-colour.glb'), join(dir, 'raw.glb'),
            (m) => ch.send('progress', { stage: 'recon', detail: m }));
        } else if (engine === 'colour') {
          ch.send('progress', { stage: 'recon', detail: 'reconstructing with colour' });
          // TripoSplat conditions on one image and does its own background removal, so
          // it takes the original front view rather than the matte
          await reconstructColour(views.front, join(dir, 'raw.glb'), {
            seed,
            onProgress: (p) => ch.send('progress', { stage: 'recon', detail: p.detail }),
          });
        } else {
          ch.send('progress', { stage: 'recon', detail: 'waking the model' });
          await reconstructViews(Object.fromEntries(Object.entries(fed).filter(([k]) => shapeViews[k])), join(dir, 'raw.glb'), {
            seed, ...quality,
            onProgress: (p) => ch.send('progress', { stage: 'recon', detail: p.detail }),
          });
        }
        const blankOpts = {
          targetHeight: heightM, targetWidth: widthM, targetDepth: depthM,
          // a carved hull only needs its voxel staircase taken off (three passes; eight
          // was tuned for the raw splat mesh and smoothed a figurine's fins away);
          // Hunyuan3D's surface net is already smooth and would only lose its creases
          smoothPasses: engine === 'colour' ? 3 : 0,
          // and it is a double wall with interior sheets, so its outer skin is rebuilt
          // from depth views before anything else touches it (lib/carve.mjs)
          carve: engine === 'colour',
          // the hybrid's mesh is clean and keeps its raw self for the dense bakes
          dense: engine === 'hybrid',
          onProgress: (p) => ch.send('progress', p),
        };
        let meta = await makeBlank(join(dir, 'raw.glb'), dir, blankOpts);
        meta.engine = engine;
        // persisted now, not only after the fit score: the score can fail and the
        // engine is what a later re-split needs to pick the right smoothing
        await writeFile(join(dir, 'blank.json'), JSON.stringify(meta, null, 2));
        await writeFile(join(dir, 'look.json'), JSON.stringify(emptyLook(meta.regions)));

        // How closely does the reconstruction actually match the photograph? Score the
        // mesh's silhouette against the matte's. It cannot be used to fix anything, but
        // it is the one honest answer to "is this a good reconstruction", and it costs
        // a few seconds against minutes of GPU time.
        meta = await finishColour(dir, meta, ch, blankOpts);
        ch.send('done', { id, ...meta });
      } catch (e) {
        ch.send('failed', { error: e.message });
      }
      return ch.end();
    }

    // Rebuild a Blank from a raw reconstruction that is already on disk. Cheap, and
    // the only way to change the real-world height or the crease threshold without
    // paying for the GPU again.
    const refine = path.match(/^\/api\/projects\/([a-z0-9-]+)\/refine$/);
    if (refine && req.method === 'POST') {
      const dir = projectDir(refine[1]);
      if (!dir) return send(res, 400, { error: 'bad id' });
      const opts = JSON.parse(await readBody(req, 4096) || '{}');
      const ch = sse(res);
      try {
        // the engine that made raw.glb decides the smoothing, same as at build time
        let prevEngine = 'shape';
        try { prevEngine = JSON.parse(await readFile(join(dir, 'blank.json'), 'utf8')).engine || 'shape'; } catch {}
        const blankOpts = {
          targetHeight: Number(opts.targetHeight) || 0.78,
          targetWidth: Number(opts.targetWidth) || null,
          targetDepth: Number(opts.targetDepth) || null,
          smoothPasses: prevEngine === 'colour' ? 3 : 0,
          carve: prevEngine === 'colour',
          dense: prevEngine === 'hybrid',
          // null, not a default: null lets makeBlank walk the crease angle down
          // until the split is useful. Forcing 38 here is what made a refine with no
          // angle collapse a smooth mesh back to a single region.
          creaseDegrees: opts.creaseDegrees ? Number(opts.creaseDegrees) : null,
          onProgress: (p) => ch.send('progress', p),
        };
        let meta = await makeBlank(join(dir, 'raw.glb'), dir, blankOpts);
        meta.engine = prevEngine;
        await writeFile(join(dir, 'blank.json'), JSON.stringify(meta, null, 2));
        await writeFile(join(dir, 'look.json'), JSON.stringify(emptyLook(meta.regions)));
        meta = await finishColour(dir, meta, ch, blankOpts);
        ch.send('done', meta);
      } catch (e) { ch.send('failed', { error: e.message }); }
      return ch.end();
    }

    const bakeM = path.match(/^\/api\/projects\/([a-z0-9-]+)\/bake$/);
    if (bakeM && req.method === 'POST') {
      const dir = projectDir(bakeM[1]);
      if (!dir) return send(res, 400, { error: 'bad id' });
      const look = JSON.parse(await readBody(req, 1 << 20));
      const meta = JSON.parse(await readFile(join(dir, 'blank.json'), 'utf8'));
      const bad = validate(look, meta.regions);
      if (bad) return send(res, 400, { error: bad });
      await writeFile(join(dir, 'look.json'), JSON.stringify(look));
      const out = await bake(dir, look, { name: 'model' });
      return send(res, 200, {
        materials: out.materials,
        glbBytes: out.glbBytes,
        usdz: !!out.usdz,
        usdzNote: out.usdzNote,
        ar: `/ar/${bakeM[1]}`,
      });
    }

    // project files
    const file = path.match(/^\/p\/([a-z0-9-]+)\/(.+)$/);
    if (file) {
      const dir = projectDir(file[1]);
      if (!dir || file[2].includes('..')) return send(res, 400, { error: 'bad path' });

      // Self-heal a Look that no longer fits its Blank. Rebuilding a blank changes the
      // region count, and any path that does so without rewriting the look leaves the
      // project dead: the studio loads, the person paints, and only the bake fails,
      // with a message about counts that means nothing to them. Checking here means
      // every route that reads a look gets a valid one.
      if (file[2] === 'look.json') {
        try {
          const meta = JSON.parse(await readFile(join(dir, 'blank.json'), 'utf8'));
          const look = JSON.parse(await readFile(join(dir, 'look.json'), 'utf8'));
          if (look.coats?.length !== meta.regions) {
            console.log(`${file[1]}: look had ${look.coats?.length} coats for ${meta.regions} regions, resetting`);
            await writeFile(join(dir, 'look.json'), JSON.stringify(emptyLook(meta.regions)));
          }
        } catch { /* no blank yet, serve whatever is there */ }
      }
      // ?dl=1 forces a download. Without it Safari hands a tapped USDZ link straight
      // to AR Quick Look, so the "download" button on an iPhone opened AR instead.
      const dl = url.searchParams.get('dl') ? file[2].split('/').pop() : null;
      return serveFile(req, res, join(dir, file[2]), { download: dl });
    }

    // the AR landing page: the thing you actually open on a phone
    const ar = path.match(/^\/ar\/([a-z0-9-]+)$/);
    if (ar) {
      const dir = projectDir(ar[1]);
      if (!dir) return send(res, 400, { error: 'bad id' });
      let meta = {};
      try { meta = JSON.parse(await readFile(join(dir, 'blank.json'), 'utf8')); } catch {}
      // A QR code for the phone, pointing at this machine's LAN address on the secure
      // port: one scan instead of typing an IP. The code is drawn server-side as SVG.
      let qr = '';
      try {
        const QR = (await import('qrcode')).default;
        const httpsPort = Number(process.env.HTTPS_PORT || PORT + 1);
        const address = lan[0] ? `https://${lan[0]}:${httpsPort}/ar/${ar[1]}` : null;
        if (address) {
          const svg = await QR.toString(address, { type: 'svg', margin: 1, color: { dark: '#e8e4da', light: '#0000' } });
          qr = `<figure class="qr"><div class="qr-svg">${svg}</div><figcaption>Scan on your phone: ${address}</figcaption></figure>`;
        }
      } catch { /* no qr module, no code: the addresses are still printed at startup */ }
      const page = (await readFile(join(ROOT, 'web', 'ar.html'), 'utf8'))
        .replace(/__QR__/g, qr)
        .replace(/__ID__/g, ar[1])
        .replace(/__HTTPS_PORT__/g, String(Number(process.env.HTTPS_PORT || PORT + 1)))
        .replace(/__DIMS__/g, meta.dimensions
          ? `${(meta.dimensions.width * 100).toFixed(0)} × ${(meta.dimensions.depth * 100).toFixed(0)} × ${(meta.dimensions.height * 100).toFixed(0)} cm`
          : 'unknown');
      return send(res, 200, page, MIME['.html']);
    }

    if (path === '/api/materials' && req.method === 'GET') {
      return send(res, 200, { weaves: WEAVES, materials: catalogue() });
    }

    // Invent a cloth. The generator behind this is the same one that made the four
    // built-ins, so a custom material is not a second-class one.
    if (path === '/api/materials' && req.method === 'POST') {
      const spec = JSON.parse(await readBody(req, 4096));
      try {
        const made = await createMaterial({
          name: spec.name,
          weave: spec.weave,
          roughness: Math.min(1, Math.max(0, Number(spec.roughness) || 0.85)),
          normalStrength: Math.min(20, Math.max(0.5, Number(spec.normalStrength) || 6)),
          threads: Math.min(120, Math.max(8, Math.round(Number(spec.threads) || 40))),
          native: /^#[0-9a-fA-F]{6}$/.test(spec.native || '') ? spec.native : '#c9ccd8',
        });
        return send(res, 200, { id: made.id, materials: catalogue() });
      } catch (e) {
        return send(res, 400, { error: e.message });
      }
    }

    const delMat = path.match(/^\/api\/materials\/([a-z0-9-]+)$/);
    if (delMat && req.method === 'DELETE') {
      await deleteMaterial(delMat[1]);
      return send(res, 200, { materials: catalogue() });
    }

    // static: the studio, its vendored libraries, and the material swatches
    // the reconstruction comparison sheet, a dev-only surface
    if (path.startsWith('/compare/') && !path.includes('..')) {
      return serveFile(req, res, join(ROOT, path.slice(1)));
    }
    if (path.startsWith('/materials/')) {
      return serveFile(req, res, join(ROOT, 'assets', 'materials', path.slice(11)), { longCache: true });
    }
    if (path.startsWith('/vendor/three/')) {
      return serveFile(req, res, join(ROOT, 'node_modules', 'three', path.slice(14)), { longCache: true });
    }
    if (path.startsWith('/vendor/model-viewer/')) {
      return serveFile(req, res, join(ROOT, 'node_modules', '@google', 'model-viewer', 'dist', path.slice(21)), { longCache: true });
    }
    const web = path === '/' ? '/index.html' : path;
    if (!web.includes('..')) return serveFile(req, res, join(ROOT, 'web', web));
    return send(res, 404, { error: 'not found' });
  } catch (e) {
    console.error(req.method, path, e);
    if (!res.headersSent) send(res, 500, { error: e.message });
    else res.end();
  }
});

await mkdir(PROJECTS, { recursive: true });
await loadCustom();
lan = Object.values(networkInterfaces()).flat()
  .filter((n) => n && n.family === 'IPv4' && !n.internal).map((n) => n.address);

server.listen(PORT, '0.0.0.0', () => {
  console.log(`photo-to-ar on http://localhost:${PORT}`);
  for (const a of lan) console.log(`  on this network: http://${a}:${PORT}   (iPhone AR works here)`);
  readiness().then((r) => console.log(r.ok ? '  GPU: ready' : `  GPU: ${r.reason}`));
});

/* A second listener over HTTPS, because Android AR does not exist without it.
 *
 * WebXR needs a secure context, and Scene Viewer's documented requirement is an https
 * model URL. Over plain http the AR button on an Android phone therefore does nothing,
 * silently, while the same page works on an iPhone through Quick Look. A self-signed
 * certificate is enough for WebXR: Chrome shows a warning once, the user accepts it,
 * and the page is a secure context from then on. It is not enough for Scene Viewer,
 * whose own fetch will refuse the certificate, which is fine: WebXR is the better mode,
 * it stays inside the browser with model-viewer's own renderer.
 *
 * Generate the pair once, with the LAN address in the SAN, from certs/san.cnf:
 *   openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout certs/key.pem  *     -out certs/cert.pem -config certs/san.cnf
 */
const HTTPS_PORT = Number(process.env.HTTPS_PORT || PORT + 1);
try {
  const [key, cert] = await Promise.all([
    readFile(join(ROOT, 'certs', 'key.pem')),
    readFile(join(ROOT, 'certs', 'cert.pem')),
  ]);
  const secure = createHttps({ key, cert }, (req, res) => server.emit('request', req, res));
  secure.listen(HTTPS_PORT, '0.0.0.0', () => {
    for (const a of lan) console.log(`  secure: https://${a}:${HTTPS_PORT}   (Android AR needs this one; accept the certificate warning once)`);
  });
} catch {
  console.log('  no certs/cert.pem: http only, so AR will work on iPhone but not Android');
}
