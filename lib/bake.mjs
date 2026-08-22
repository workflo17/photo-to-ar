/**
 * Blank + Look, in. A glTF and a USDZ that agree with each other, out.
 *
 * "Agree with each other" is the whole job. Android reads the glTF through Scene
 * Viewer and iOS reads the USDZ through AR Quick Look, and if the two files disagree
 * about scale, colour space or which face wears which material, the same product
 * looks like two different products depending on the shopper's phone.
 *
 * Regions wearing identical coats are merged into one material before anything is
 * written, because both formats pay per material and AR Quick Look in particular
 * slows down with a long material list.
 */
import { Document, NodeIO } from '@gltf-transform/core';
import { tangents, unweld, weld } from '@gltf-transform/functions';
import { generateTangents } from 'mikktspace';
import { KHRMaterialsSheen, KHRDracoMeshCompression } from '@gltf-transform/extensions';
import { draco } from '@gltf-transform/functions';
import draco3d from 'draco3dgltf';
import { readFile, writeFile } from 'node:fs/promises';
import sharp from 'sharp';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { groups, tintToLinear } from './look.mjs';
import { getMaterial } from './materials.mjs';

const run = promisify(execFile);
const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const MATERIALS_DIR = join(ROOT, 'assets', 'materials');

/**
 * Pull one group's triangles out into standalone vertex arrays.
 *
 * Splitting rather than using glTF primitive offsets keeps every group's UVs
 * independently scalable, and independent UVs are the only portable way to give two
 * parts different texture scales: USD has no equivalent of KHR_texture_transform that
 * AR Quick Look honours reliably.
 */
function extract(src, triangleIds, uvScale, atlasUVs) {
  const { positions, normals, uvs, indices } = src;

  // The photo-colour path cannot share vertices. Its UVs come from a cube atlas that
  // is unwelded by construction, so a vertex sitting on a tile seam needs a different
  // UV in each tile. Emit three fresh vertices per triangle and read the UV straight
  // out of the per-triangle atlas.
  if (atlasUVs) {
    const pos = [], nrm = [], uv = [], idx = [];
    for (const t of triangleIds) {
      for (let k = 0; k < 3; k++) {
        const v = indices[t * 3 + k];
        pos.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
        nrm.push(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]);
        uv.push(atlasUVs[t * 6 + k * 2], atlasUVs[t * 6 + k * 2 + 1]);
        idx.push(idx.length);
      }
    }
    return {
      positions: new Float32Array(pos), normals: new Float32Array(nrm),
      uvs: new Float32Array(uv), indices: new Uint32Array(idx),
    };
  }

  const remap = new Map();
  const pos = [], nrm = [], uv = [], idx = [];
  for (const t of triangleIds) {
    for (let k = 0; k < 3; k++) {
      const v = indices[t * 3 + k];
      let n = remap.get(v);
      if (n === undefined) {
        n = pos.length / 3;
        remap.set(v, n);
        pos.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
        nrm.push(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]);
        uv.push(uvs[v * 2] * uvScale, uvs[v * 2 + 1] * uvScale);
      }
      idx.push(n);
    }
  }
  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(nrm),
    uvs: new Float32Array(uv),
    indices: new Uint32Array(idx),
  };
}

/** Read the Blank's single primitive into plain typed arrays. */
async function readBlank(dir) {
  const doc = await new NodeIO().read(join(dir, 'blank.glb'));
  const prim = doc.getRoot().listMeshes().flatMap((m) => m.listPrimitives())[0];
  if (!prim) throw new Error('blank.glb has no primitive');
  const regions = JSON.parse(await readFile(join(dir, 'regions.json'), 'utf8'));
  const perTriangle = new Uint16Array(Buffer.from(regions.perTriangle, 'base64').buffer);
  let atlasUVs = null, hasDetail = false, hasGlow = false;
  try {
    const raw = await readFile(join(dir, 'atlas.bin'));
    atlasUVs = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const meta = JSON.parse(await readFile(join(dir, 'blank.json'), 'utf8'));
    hasDetail = !!meta.hasDetail;
    hasGlow = !!meta.hasGlow;
  } catch { /* no colour on this blank */ }
  return {
    atlasUVs,
    hasDetail,
    hasGlow,
    positions: prim.getAttribute('POSITION').getArray(),
    normals: prim.getAttribute('NORMAL').getArray(),
    uvs: prim.getAttribute('TEXCOORD_0').getArray(),
    indices: prim.getIndices().getArray(),
    perTriangle,
    regions: regions.regions,
  };
}

/**
 * Bake a Look onto a Blank.
 *
 * Returns the paths written and the material count, so a caller can tell the person
 * what actually shipped rather than just "done".
 */
export async function bake(dir, look, { name = 'model', onProgress } = {}) {
  const log = (m) => onProgress?.({ stage: 'bake', detail: m });
  const src = await readBlank(dir);

  // regions -> triangles, once, so each group is a single pass
  const trisByRegion = new Map();
  for (let t = 0; t < src.perTriangle.length; t++) {
    const r = src.perTriangle[t];
    let list = trisByRegion.get(r);
    if (!list) trisByRegion.set(r, list = []);
    list.push(t);
  }

  const gs = groups(look);
  log(`${look.coats.length} regions collapse to ${gs.length} material${gs.length === 1 ? '' : 's'}`);

  const doc = new Document();
  const buffer = doc.createBuffer();
  const scene = doc.createScene();
  const mesh = doc.createMesh(name);
  const sheenExt = doc.createExtension(KHRMaterialsSheen);

  // One texture object per file, shared across groups that use the same map.
  //
  // Normals get halved on the way in. They arrive as 1024 PNGs at about 2.5 MB each,
  // which on a two-fabric model is two thirds of the whole file, and AR downloads the
  // entire thing before it shows anything. 512 is the same resolution the USDZ side
  // has always used, so this also stops the two formats disagreeing about weight.
  const texCache = new Map();
  const texture = async (file) => {
    if (texCache.has(file)) return texCache.get(file);
    const path = join(MATERIALS_DIR, file);
    const isNormal = /_normal\.png$/.test(file);
    const bytes = isNormal
      ? await sharp(path).resize(512, 512, { kernel: 'lanczos3' }).png({ compressionLevel: 9 }).toBuffer()
      : await readFile(path);
    const tex = doc.createTexture(file)
      .setImage(bytes)
      .setMimeType(file.endsWith('.png') ? 'image/png' : 'image/jpeg');
    texCache.set(file, tex);
    return tex;
  };

  // what the USDZ builder will need: the same split, as plain JSON
  const usdParts = [];
  let photoTexture = null;

  for (const [i, g] of gs.entries()) {
    const tris = g.regions.flatMap((r) => trisByRegion.get(r) || []);
    if (!tris.length) continue;
    const def = getMaterial(g.coat.materialId);
    const isPhoto = def.kind === 'photo';
    if (isPhoto && !src.atlasUVs) {
      log('a part asks for the photo colour but this blank has none, falling back to satin');
    }
    const usePhoto = isPhoto && src.atlasUVs;
    const part = extract(src, tris, 1 / g.coat.tileScale, usePhoto ? src.atlasUVs : null);
    const tint = tintToLinear(g.coat.tint);

    const prim = doc.createPrimitive()
      .setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(part.positions).setBuffer(buffer))
      .setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(part.normals).setBuffer(buffer))
      .setAttribute('TEXCOORD_0', doc.createAccessor().setType('VEC2').setArray(part.uvs).setBuffer(buffer))
      .setIndices(doc.createAccessor().setType('SCALAR').setArray(part.indices).setBuffer(buffer));

    const matName = `${def.name}${gs.length > 1 ? ` ${i + 1}` : ''}`;
    const mat = doc.createMaterial(matName)
      .setBaseColorFactor([...tint, 1])
      .setRoughnessFactor(g.coat.roughness ?? def.roughness ?? 0.6)
      .setMetallicFactor(g.coat.metallic ?? def.metallic ?? 0);

    if (usePhoto) {
      // one texture, straight off disk, and no tint multiply by default: the point of
      // this material is that the colour is already the product's own
      const bytes = await readFile(join(dir, 'photo.jpg'));
      const tex = doc.createTexture('photo').setImage(bytes).setMimeType('image/jpeg');
      mat.setBaseColorTexture(tex);
      // the detail map: the raw mesh's surface relief, baked on the same atlas
      if (src.hasDetail) {
        const nb = await readFile(join(dir, 'detail.png'));
        mat.setNormalTexture(doc.createTexture('detail').setImage(nb).setMimeType('image/png'));
      }
      // the photographed glow as emission, unless this coat turned it off
      const useGlow = src.hasGlow && g.coat.glow !== false;
      if (useGlow) {
        const eb = await readFile(join(dir, 'glow.jpg'));
        mat.setEmissiveTexture(doc.createTexture('glow').setImage(eb).setMimeType('image/jpeg'))
          .setEmissiveFactor([1, 1, 1]);
      }
      // a splat mesh is a soup of thin patches whose winding cannot be made globally
      // consistent; a face that still points inward must not become a hole
      mat.setDoubleSided(true);
      photoTexture = 'photo.jpg';
    } else if (def.kind === 'textured') {
      mat.setBaseColorTexture(await texture(def.basecolor));
      mat.setMetallicRoughnessTexture(await texture(def.orm));
      mat.setOcclusionTexture(await texture(def.orm));
      mat.setNormalTexture(await texture(def.normal));
      // the ORM convention: occlusion in R, roughness in G, metalness in B. Without
      // this the roughness map is read from the red channel and everything looks wet.
      mat.setOcclusionStrength(1);
      if (def.sheen) {
        mat.setExtension('KHR_materials_sheen', sheenExt.createSheen()
          .setSheenColorFactor([1, 1, 1])
          .setSheenRoughnessFactor(def.sheen));
      }
    }
    prim.setMaterial(mat);
    mesh.addPrimitive(prim);

    usdParts.push({
      material: matName,
      kind: def.kind,
      tint,
      roughness: g.coat.roughness ?? def.roughness ?? 0.6,
      metallic: g.coat.metallic ?? def.metallic ?? 0,
      textures: usePhoto
        ? { basecolor: '__photo__', normal: src.hasDetail ? '__detail__' : null, orm: null,
            emissive: src.hasGlow && g.coat.glow !== false ? '__glow__' : null }
        : (def.kind === 'textured'
          ? { basecolor: def.basecolor, normal: def.normal, orm: def.orm } : null),
      positions: Array.from(part.positions),
      normals: Array.from(part.normals),
      uvs: Array.from(part.uvs),
      indices: Array.from(part.indices),
    });
  }

  // Pin the tangent basis rather than letting three separate renderers derive one.
  //
  // Normally deriving is fine. It is not fine here: the UVs come from a box
  // projection, which mirrors on opposite faces and seams at every axis crossing, and
  // three.js, Scene Viewer and RealityKit each resolve mirrored/seamed frames
  // differently. Without an authored basis the same fabric catches light three
  // different ways on three surfaces of the same chair. Costs 16 bytes a vertex.
  // mikktspace works per-face, so the pass refuses an indexed primitive. Unweld,
  // generate, weld back: the weld re-merges only vertices whose tangents also agree,
  // which is exactly the behaviour wanted at a mirrored UV seam.
  await doc.transform(unweld(), tangents({ generateTangents }), weld());

  scene.addChild(doc.createNode(name).setMesh(mesh));
  const glbPath = join(dir, `${name}.glb`);
  await new NodeIO().registerExtensions([KHRMaterialsSheen]).write(glbPath, doc);
  const glbBytes = (await readFile(glbPath)).length;
  log(`glTF written, ${(glbBytes / 1e6).toFixed(2)} MB`);

  // A Draco-compressed twin for the AR page. Geometry is about a third of the plain
  // file and compresses about 8:1; the textures do not shrink, so the gain is what it
  // is. The plain file stays for viewers without a decoder (the showcase page, which
  // cannot load one under its CSP). Best effort: the plain file is the product.
  let liteBytes = null;
  try {
    const encoder = await draco3d.createEncoderModule();
    const lite = await new NodeIO().registerExtensions([KHRMaterialsSheen, KHRDracoMeshCompression])
      .registerDependencies({ 'draco3d.encoder': encoder }).read(glbPath);
    await lite.transform(draco({ method: 'edgebreaker' }));
    await new NodeIO().registerExtensions([KHRMaterialsSheen, KHRDracoMeshCompression])
      .registerDependencies({ 'draco3d.encoder': encoder }).write(join(dir, `${name}-lite.glb`), lite);
    liteBytes = (await readFile(join(dir, `${name}-lite.glb`))).length;
    log(`Draco twin for AR, ${(liteBytes / 1e6).toFixed(2)} MB`);
  } catch (e) {
    log(`Draco twin skipped: ${(e.message || '').split('\n')[0]}`);
  }

  // ---- the iOS half -------------------------------------------------------
  const partsPath = join(dir, '_usdparts.json');
  await writeFile(partsPath, JSON.stringify({ name, parts: usdParts, photoDir: photoTexture ? dir : null }));
  let usdz = null, usdzNote = null;
  try {
    const { stdout } = await run('python', [join(ROOT, 'tools', 'build_usdz.py'), partsPath, dir],
      { maxBuffer: 1 << 26 });
    usdz = join(dir, `${name}.usdz`);
    log(`USDZ written${/valid/i.test(stdout) ? ', validated' : ''}`);
  } catch (e) {
    // A missing USDZ is a real loss (no iOS AR) but not a reason to throw away a
    // perfectly good glTF, so it is reported rather than raised.
    usdzNote = (e.stderr || e.message || '').trim().split('\n').slice(-3).join(' ');
    log(`USDZ failed: ${usdzNote}`);
  }

  return { glb: glbPath, usdz, usdzNote, materials: gs.length, glbBytes, liteBytes };
}
