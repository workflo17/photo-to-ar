/**
 * Photograph in, raw geometry out.
 *
 * Hunyuan3D 2.0 running locally through ComfyUI's native nodes: the image is encoded
 * by the checkpoint's own vision tower, sampled as a shape latent, decoded to a voxel
 * field, and marching-cubed into a mesh.
 *
 * The interface is one function, because everything a caller needs to decide is a
 * photograph and a seed. Behind it sit the upload, the graph, the polling, the output
 * hunt and the failure modes. Callers never learn what a Hunyuan3Dv2Conditioning node
 * is, and the test fake never has to pretend to be one.
 *
 * What this step does NOT produce is texture or UVs. ComfyUI's native path is shape
 * only. That is the honest division of labour in this app: the model reconstructs
 * form, the studio owns the surface.
 */
import { writeFile } from 'node:fs/promises';
import { alive, run, outputsOf, fetchOutput, upload, objectInfo } from './comfy.mjs';

const CKPT = process.env.HUNYUAN_CKPT || 'hunyuan3d-dit-v2.safetensors';
// The multi-view variant. The single-view checkpoint was never trained on the
// concatenated multi-view conditioning layout, so feeding it extra views is
// accepted and ignored: measured A/B on this machine, 1 view scored IoU 0.4347 and
// 2 views 0.4259 through the single-view weights. Matched by suffix because the
// file may live behind a junction and carry a directory prefix.
const CKPT_MV_SUFFIX = 'hunyuan3d-dit-v2-mv.safetensors';

/** Everything that must be true before a reconstruction can be attempted. */
export async function readiness() {
  if (!await alive()) {
    return { ok: false, reason: 'ComfyUI is not answering on its port. Start it and retry.' };
  }
  let info;
  try { info = await objectInfo(); }
  catch (e) { return { ok: false, reason: `ComfyUI answered but /object_info failed: ${e.message}` }; }

  const missing = ['ImageOnlyCheckpointLoader', 'CLIPVisionEncode', 'Hunyuan3Dv2Conditioning',
    'EmptyLatentHunyuan3Dv2', 'KSampler', 'VAEDecodeHunyuan3D', 'VoxelToMesh', 'SaveGLB']
    .filter((n) => !info[n]);
  if (missing.length) return { ok: false, reason: `ComfyUI is missing nodes: ${missing.join(', ')}` };

  const spec = info.ImageOnlyCheckpointLoader?.input?.required?.ckpt_name;
  const have = Array.isArray(spec?.[0]) ? spec[0] : (spec?.[1]?.options || []);
  if (!have.includes(CKPT)) {
    return { ok: false, reason: `${CKPT} is not in ComfyUI's checkpoints. Found: ${have.join(', ') || 'none'}` };
  }
  const mvCheckpoint = have.find((n) => n.endsWith(CKPT_MV_SUFFIX)) || null;
  return { ok: true, checkpoint: CKPT, mvCheckpoint, info };
}

/**
 * Reconstruct one image into a glb on disk.
 *
 * resolution is the voxel grid the shape latent decodes into. 3072 is what the
 * case study used; lower is faster and blockier. The 3080 holds 3072 comfortably
 * for a single batch, and OOMs are far more likely to come from another model still
 * resident in VRAM than from this number.
 */
/**
 * Reconstruct from one to four views.
 *
 * `views` is {front, left, back, right} with at least `front` set. Anything missing is
 * simply not conditioned on, which the multi-view node supports: all four of its inputs
 * are optional.
 *
 * This is the single largest quality lever in the app. From one photograph the model
 * cannot see the back of the object, so it invents one, smoothly and confidently, and
 * no sampler or grid resolution repairs an invention. Two views already fix most of it.
 */
export async function reconstructViews(views, outPath, opts = {}) {
  const named = Object.entries(views).filter(([, b]) => b && b.length > 1024);
  if (!named.length) throw new Error('no views supplied');
  if (named.length === 1) return reconstruct(named[0][1], outPath, opts);
  return reconstruct(named[0][1], outPath, { ...opts, extraViews: Object.fromEntries(named) });
}

export async function reconstruct(imageBuffer, outPath, {
  seed = 7, steps = 50, cfg = 5.0, resolution = 3072,
  // The two dials that decide how the isosurface is extracted, which is where most
  // of the visible quality lives. "surface net" places vertices by where the field
  // actually crosses the threshold rather than on grid corners, so it does not
  // produce the stair-stepping plain marching cubes gives you on a smooth form.
  algorithm = 'surface net', octreeResolution = 512, threshold = 0.6,
  // {front, left, back, right} of image buffers. When present the graph swaps to
  // multi-view conditioning, which uses the same checkpoint.
  extraViews = null,
  onProgress,
} = {}) {
  const ready = await readiness();
  if (!ready.ok) throw new Error(ready.reason);
  const info = ready.info;

  // pick samplers this build actually ships rather than assuming names
  const schedulers = info.KSampler.input.required.scheduler[0];
  const scheduler = ['normal', 'simple', 'karras'].find((s) => schedulers.includes(s));
  const samplers = info.KSampler.input.required.sampler_name[0];
  const sampler = ['euler', 'euler_ancestral'].find((s) => samplers.includes(s));
  if (!sampler || !scheduler) throw new Error('no compatible sampler/scheduler in this ComfyUI');

  // Each view needs its own LoadImage and its own CLIPVisionEncode; the multi-view
  // node then takes one encoded output per side. Node ids above 100 are the extra
  // views so the fixed part of the graph keeps its familiar numbering.
  const workflow = {
    2: { class_type: 'ImageOnlyCheckpointLoader', inputs: { ckpt_name: CKPT } },
  };

  let conditioning;
  if (extraViews) {
    // the mv weights, or an honest downgrade: without them the extra views would be
    // silently ignored, which is worse than saying so
    var mvCkpt = ready.mvCheckpoint;
    if (!mvCkpt) {
      onProgress?.({ stage: 'recon', detail: 'multi-view weights not installed, using the front view only' });
      const front = Object.values(extraViews)[0];
      return reconstruct(front, outPath, { seed, steps, cfg, resolution, algorithm, octreeResolution, threshold, onProgress });
    }
    workflow[2] = { class_type: 'ImageOnlyCheckpointLoader', inputs: { ckpt_name: mvCkpt } };
    const sides = {};
    let id = 100;
    for (const [side, buf] of Object.entries(extraViews)) {
      const uploaded = await upload(buf, `p2ar-${side}-${Date.now()}.png`);
      workflow[id] = { class_type: 'LoadImage', inputs: { image: uploaded } };
      workflow[id + 1] = {
        class_type: 'CLIPVisionEncode',
        inputs: { clip_vision: ['2', 1], image: [String(id), 0], crop: 'center' },
      };
      sides[side] = [String(id + 1), 0];
      id += 2;
      onProgress?.({ stage: 'uploaded', detail: `${side} view` });
    }
    workflow[4] = { class_type: 'Hunyuan3Dv2ConditioningMultiView', inputs: sides };
    conditioning = Object.keys(sides).join(' + ') + ' (mv weights)';
  } else {
    const name = await upload(imageBuffer, `photo-to-ar-${Date.now()}.png`);
    onProgress?.({ stage: 'uploaded', detail: name });
    workflow[1] = { class_type: 'LoadImage', inputs: { image: name } };
    workflow[3] = { class_type: 'CLIPVisionEncode', inputs: { clip_vision: ['2', 1], image: ['1', 0], crop: 'center' } };
    workflow[4] = { class_type: 'Hunyuan3Dv2Conditioning', inputs: { clip_vision_output: ['3', 0] } };
    conditioning = 'single view';
  }

  Object.assign(workflow, {
    5: { class_type: 'EmptyLatentHunyuan3Dv2', inputs: { resolution, batch_size: 1 } },
    6: {
      class_type: 'KSampler',
      inputs: {
        model: ['2', 0], positive: ['4', 0], negative: ['4', 1], latent_image: ['5', 0],
        seed, steps, cfg, sampler_name: sampler, scheduler, denoise: 1.0,
      },
    },
    // octree resolution is the detail dial: 256 trades a readable silhouette against
    // a mesh small enough to hand to a phone
    7: { class_type: 'VAEDecodeHunyuan3D', inputs: { samples: ['6', 0], vae: ['2', 2], num_chunks: 8000, octree_resolution: octreeResolution } },
    // The decode returns a VOXEL, not a mesh. Isosurface extraction happens here, and
    // SaveGLB will reject the graph outright without it.
    8: { class_type: 'VoxelToMesh', inputs: { voxel: ['7', 0], algorithm, threshold } },
    9: { class_type: 'SaveGLB', inputs: { mesh: ['8', 0], filename_prefix: 'photo-to-ar/recon' } },
  });

  onProgress?.({ stage: 'sampling', detail: `${conditioning}, ${steps} steps, ${algorithm} at ${octreeResolution}` });
  const entry = await run(workflow, {
    label: 'reconstruction',
    onProgress: (q) => onProgress?.({ stage: 'sampling', detail: `${q.running} running, ${q.pending} queued` }),
  });

  const glb = outputsOf(entry).find((f) => /\.glb$/i.test(f.filename));
  if (!glb) {
    const got = outputsOf(entry).map((f) => f.filename).join(', ') || 'nothing';
    throw new Error(`the graph finished but produced no glb (outputs: ${got})`);
  }

  const buf = await fetchOutput(glb);
  await writeFile(outPath, buf);
  onProgress?.({ stage: 'done', detail: `${(buf.length / 1e6).toFixed(1)} MB` });
  return { path: outPath, bytes: buf.length };
}
