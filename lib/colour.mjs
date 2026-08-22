/**
 * Photograph in, a mesh that is the product's own colour out.
 *
 * TripoSplat reconstructs the object as a gaussian splat and meshes it, and unlike the
 * Hunyuan3D path the result carries colour: it recovers cream boucle on the shell and
 * walnut on the legs as different colours, which is a per-part distinction the
 * procedural fabric library cannot express at all.
 *
 * It also replaces the matte. BiRefNet is a trained segmentation model rather than the
 * flood fill in matte.py, so it is the one route here that does not fall over on a
 * product photographed in a room instead of against a seamless.
 *
 * Measured on this machine: 6.0 to 6.4 GB peak of 10.0, about 30 seconds.
 *
 * WHAT THIS DOES NOT PRODUCE is a texture. `SplatToMesh` emits per-vertex colours on an
 * unlit material with no UVs and no images, and AR Quick Look reads neither vertex
 * colours nor displayColor, so a USDZ built straight from this renders plain grey on an
 * iPhone with no error anywhere. The colour has to be rasterised into a real baseColor
 * map through the Blank's UVs before it can ship. lib/vertexbake.mjs does that.
 */
import { writeFile } from 'node:fs/promises';
import { alive, run, outputsOf, fetchOutput, upload, objectInfo } from './comfy.mjs';

const NEEDED = [
  'LoadBackgroundRemovalModel', 'RemoveBackground', 'TripoSplatPreprocessImage',
  'CLIPVisionLoader', 'VAELoader', 'TripoSplatConditioning', 'UNETLoader',
  'KSampler', 'VAEDecodeTripoSplat', 'SplatToMesh', 'SaveGLB',
];

const WEIGHTS = {
  unet: 'triposplat_fp16.safetensors',
  decoder: 'triposplat_vae_decoder_fp16.safetensors',
  imageVae: 'flux2-vae.safetensors',
  clipVision: 'dino_v3_vit_h.safetensors',
};

/** Everything that must be true before a coloured reconstruction can be attempted. */
export async function readiness() {
  if (!await alive()) {
    return { ok: false, reason: 'ComfyUI is not answering on its port.' };
  }
  const info = await objectInfo();
  const missing = NEEDED.filter((n) => !info[n]);
  if (missing.length) return { ok: false, reason: `missing nodes: ${missing.join(', ')}` };

  // ComfyUI describes a dropdown two different ways depending on the node's vintage:
  // older ones put the option list in slot 0, newer ones put the literal string "COMBO"
  // there and the options under .options in slot 1. Reading only slot 0 turns "COMBO"
  // into the character 'C' and the graph is rejected for a value not in the list.
  const optionsOf = (spec) => {
    if (!spec) return [];
    if (Array.isArray(spec[0])) return spec[0];
    return spec[1]?.options || [];
  };
  const have = {
    unet: optionsOf(info.UNETLoader?.input?.required?.unet_name),
    vae: optionsOf(info.VAELoader?.input?.required?.vae_name),
    clip: optionsOf(info.CLIPVisionLoader?.input?.required?.clip_name),
    bg: optionsOf(info.LoadBackgroundRemovalModel?.input?.required?.bg_removal_name),
  };
  const absent = [];
  if (!have.unet.includes(WEIGHTS.unet)) absent.push(WEIGHTS.unet);
  if (!have.vae.includes(WEIGHTS.decoder)) absent.push(WEIGHTS.decoder);
  if (!have.vae.includes(WEIGHTS.imageVae)) absent.push(WEIGHTS.imageVae);
  if (!have.clip.includes(WEIGHTS.clipVision)) absent.push(WEIGHTS.clipVision);
  if (!have.bg.length) absent.push('a background removal model');
  if (absent.length) return { ok: false, reason: `missing weights: ${absent.join(', ')}` };

  return { ok: true, info, bgModel: have.bg[0] };
}

/**
 * Reconstruct a coloured mesh from one photograph.
 *
 * `resolution` is the marching grid SplatToMesh runs over, not the splat count. 384 is
 * its own default and produces roughly two million triangles, which the finish pass
 * then simplifies; the number that matters for the app is the triangle budget there,
 * not here.
 */
export async function reconstructColour(imageBuffer, outPath, {
  seed = 7, steps = 20, cfg = 3.0, gaussians = 262144,
  resolution = 384, colorSharpen = 2.0, smooth = 0, onProgress,
} = {}) {
  const ready = await readiness();
  if (!ready.ok) throw new Error(ready.reason);

  const name = await upload(imageBuffer, `p2ar-colour-${Date.now()}.png`);
  onProgress?.({ stage: 'colour', detail: 'uploaded, removing the background' });

  const workflow = {
    1: { class_type: 'LoadImage', inputs: { image: name } },
    2: { class_type: 'LoadBackgroundRemovalModel', inputs: { bg_removal_name: ready.bgModel } },
    3: { class_type: 'RemoveBackground', inputs: { bg_removal_model: ['2', 0], image: ['1', 0] } },
    // the mask is eroded a little before it conditions anything, because a matte edge
    // that still carries a rim of backdrop pulls that colour into the reconstruction
    4: { class_type: 'TripoSplatPreprocessImage', inputs: { image: ['1', 0], mask: ['3', 0], erode_radius: 1, size: 1024 } },
    5: { class_type: 'CLIPVisionLoader', inputs: { clip_name: WEIGHTS.clipVision } },
    6: { class_type: 'VAELoader', inputs: { vae_name: WEIGHTS.imageVae } },
    7: { class_type: 'TripoSplatConditioning', inputs: { clip_vision: ['5', 0], vae: ['6', 0], image: ['4', 0] } },
    8: { class_type: 'UNETLoader', inputs: { unet_name: WEIGHTS.unet, weight_dtype: 'default' } },
    9: {
      class_type: 'KSampler',
      inputs: {
        model: ['8', 0], positive: ['7', 0], negative: ['7', 1], latent_image: ['7', 2],
        seed, steps, cfg, sampler_name: 'dpmpp_2m', scheduler: 'simple', denoise: 1.0,
      },
    },
    10: { class_type: 'VAELoader', inputs: { vae_name: WEIGHTS.decoder } },
    11: { class_type: 'VAEDecodeTripoSplat', inputs: { samples: ['9', 0], vae: ['10', 0], num_gaussians: gaussians, seed } },
    12: {
      class_type: 'SplatToMesh',
      inputs: {
        splat: ['11', 0], resolution, kernel: 5, smooth, level: 0.4,
        min_component: 500, min_opacity: 0.02, color_sharpen: colorSharpen,
      },
    },
    13: { class_type: 'SaveGLB', inputs: { mesh: ['12', 0], filename_prefix: 'photo-to-ar/colour' } },
  };

  onProgress?.({ stage: 'colour', detail: `${steps} steps, ${gaussians.toLocaleString()} gaussians` });
  const entry = await run(workflow, {
    label: 'triposplat',
    onProgress: (q) => onProgress?.({ stage: 'colour', detail: `${q.running} running, ${q.pending} queued` }),
  });

  const glb = outputsOf(entry).find((f) => /\.glb$/i.test(f.filename));
  if (!glb) {
    const got = outputsOf(entry).map((f) => f.filename).join(', ') || 'nothing';
    throw new Error(`the graph finished but produced no glb (outputs: ${got})`);
  }
  const buf = await fetchOutput(glb);
  await writeFile(outPath, buf);
  onProgress?.({ stage: 'colour', detail: `${(buf.length / 1e6).toFixed(1)} MB coloured mesh` });
  return { path: outPath, bytes: buf.length };
}

/**
 * Just the subject mask, from the same segmentation model the reconstruction uses.
 *
 * The colour flood-fill matte cannot separate a figure from a studio gradient (it
 * kept 77% of the Shin Godzilla frame as subject); BiRefNet can, and the projection
 * of a photograph onto the model needs that edge to be right, because every pixel
 * of backdrop that slips through lands on the model as a smear of grey.
 *
 * Returns a PNG buffer: white where the subject is.
 */
export async function subjectMask(imageBuffer) {
  const ready = await readiness();
  if (!ready.ok) throw new Error(ready.reason);
  const name = await upload(imageBuffer, `p2ar-mask-${Date.now()}.png`);
  const workflow = {
    1: { class_type: 'LoadImage', inputs: { image: name } },
    2: { class_type: 'LoadBackgroundRemovalModel', inputs: { bg_removal_name: ready.bgModel } },
    3: { class_type: 'RemoveBackground', inputs: { bg_removal_model: ['2', 0], image: ['1', 0] } },
    4: { class_type: 'MaskToImage', inputs: { mask: ['3', 0] } },
    5: { class_type: 'SaveImage', inputs: { images: ['4', 0], filename_prefix: 'photo-to-ar/mask' } },
  };
  const entry = await run(workflow, { label: 'mask' });
  const png = outputsOf(entry).find((f) => /\.png$/i.test(f.filename));
  if (!png) throw new Error('the mask graph produced no image');
  return fetchOutput(png);
}
