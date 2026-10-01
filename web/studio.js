/**
 * The studio: a Blank on the left, its Look on the right.
 *
 * The one structural idea: the index buffer is sorted by region once at load, so
 * every region owns a contiguous run of triangles and becomes a draw group with its
 * own material slot. Changing what a region is wearing is then a material swap in an
 * array, not a geometry rebuild, which is what keeps the colour picker feeling live
 * on a sixty-thousand triangle mesh.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const $ = (s) => document.querySelector(s);
const state = {
  id: null, look: null, regions: [], selected: null,
  materials: [], meshMats: [], mesh: null, catalogue: [], dirty: false,
};

/* ------------------------------------------------------------------ scene */
const canvas = $('#view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
// A software GL (SwiftShader in headless Edge, llvmpipe on a server) renders the
// studio, but a 180k-vertex mesh with a normal map takes it minutes per frame and
// the test harness's screenshot never returns. Detect it and keep the preview light
// there: no relief map, one pixel per CSS pixel. A real GPU gets everything.
const glInfo = renderer.getContext().getExtension('WEBGL_debug_renderer_info');
const glName = glInfo ? renderer.getContext().getParameter(glInfo.UNMASKED_RENDERER_WEBGL) : '';
const softwareGL = /swiftshader|llvmpipe|software/i.test(glName);
if (softwareGL) renderer.setPixelRatio(1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
camera.position.set(1.1, 0.85, 1.35);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 0.38, 0);

/* Lighting is calibrated to what the AR viewers do, not to what flatters a swatch.
   Scene Viewer and AR Quick Look light a model with a neutral environment whose
   irradiance is about one, so an albedo maps to roughly itself. The old rig here
   (hemisphere 1.5 + key 2.4 + rim 1.0) put about 3.5x that on a lit face: a cream
   chair went chalk white and a near-black figurine, every vertex under 0.05
   luminance, rendered mid grey. RoomEnvironment is procedural, so it costs no bytes,
   and the two directional lights are kept low for shape, not for exposure. */
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.9;
const key = new THREE.DirectionalLight(0xfff6ec, 0.9);
key.position.set(2.4, 3.2, 2.0);
scene.add(key);
const rim = new THREE.DirectionalLight(0x9bb0ff, 0.35);
rim.position.set(-2.2, 1.4, -1.8);
scene.add(rim);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(1, 64).rotateX(-Math.PI / 2),   // scaled to the object at load
  new THREE.MeshStandardMaterial({ color: 0x0b0d16, roughness: 1, metalness: 0 }),
);
scene.add(ground);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

/* --------------------------------------------------------------- textures */
const texLoader = new THREE.TextureLoader();
const texCache = new Map();
function tex(file, srgb) {
  const k = file + (srgb ? '|s' : '');
  if (texCache.has(k)) return texCache.get(k);
  const t = texLoader.load(`/materials/${file}`);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  texCache.set(k, t);
  return t;
}

/** Build the three.js material one coat describes. */
function materialFor(coat) {
  const def = state.catalogue.find((m) => m.id === coat.materialId) || state.catalogue[0];
  const m = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(coat.tint),
    roughness: coat.roughness ?? def.roughness ?? 0.6,
    metalness: coat.metallic ?? def.metallic ?? 0,
  });
  if (def.kind === 'photo') {
    // Preview straight from COLOR_0, which the blank already carries. That is the same
    // data the exporter rasterises into the atlas, so what the studio shows and what
    // ships agree without the browser needing the atlas UVs at all.
    m.vertexColors = true;
    if (state.atlasUV) {
      // the shipped texture itself, through the atlas UVs: what the studio shows is
      // then the file, texel for texel, and the texture carries the dense mesh's
      // colour detail that the vertex colours never had
      const map = new THREE.TextureLoader().load(`/p/${state.id}/photo.jpg`);
      map.flipY = false;   // the atlas is written in glTF's top-left convention
      map.colorSpace = THREE.SRGBColorSpace;
      map.channel = 1;
      m.map = map;
      m.vertexColors = false;
    }
    m.side = THREE.DoubleSide;   // matches the export: no holes from inward faces
    m.color = new THREE.Color(coat.tint || '#ffffff');
    m.roughness = coat.roughness ?? def.roughness ?? 0.9;
    m.metalness = coat.metallic ?? 0;
    // the photograph's own highlights are already in the colour; a rendered sheen
    // on top reads as grey on dark products
    m.specularIntensity = 0.25;
    if (state.hasDetail) {
      // the baked relief, read through the atlas UVs carried as uv1
      const nm = new THREE.TextureLoader().load(`/p/${state.id}/detail.png`);
      nm.flipY = false;
      nm.colorSpace = THREE.NoColorSpace;
      nm.channel = 1;
      m.normalMap = nm;
    }
    if (state.hasGlow && coat.glow !== false) {
      const em = new THREE.TextureLoader().load(`/p/${state.id}/glow.jpg`);
      em.flipY = false;
      em.colorSpace = THREE.SRGBColorSpace;
      em.channel = 1;
      m.emissiveMap = em;
      m.emissive = new THREE.Color(0xffffff);
      m.emissiveIntensity = 1.2;
    }
    return m;
  }
  if (def.kind === 'textured') {
    const s = 1 / coat.tileScale;
    const set = (map, srgb) => {
      const t = tex(map, srgb).clone();
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(s, s);
      t.needsUpdate = true;
      return t;
    };
    m.map = set(`${def.id}_weave.jpg`, true);
    m.normalMap = set(`${def.id}_normal.png`, false);
    // ORM packs occlusion in R, roughness in G, metalness in B. three.js reads
    // roughness from G and metalness from B off the same texture, which is exactly
    // the convention these maps were authored to.
    const orm = set(`${def.id}_orm.jpg`, false);
    m.roughnessMap = orm;
    m.metalnessMap = orm;
    m.aoMap = orm;
    if (def.sheen) { m.sheen = 1; m.sheenRoughness = def.sheen; m.sheenColor = new THREE.Color(0xffffff); }
    if (coat.roughness === null) m.roughness = 1;   // let the map drive it
  }
  return m;
}

/* ----------------------------------------------------------------- loading */
async function load(id) {
  state.id = id;
  const [gltf, regions, look] = await Promise.all([
    new GLTFLoader().loadAsync(`/p/${id}/blank.glb`),
    fetch(`/p/${id}/regions.json`).then((r) => r.json()),
    fetch(`/p/${id}/look.json`).then((r) => r.json()),
  ]);
  state.regions = regions.regions;
  state.look = look;
  const meta = await fetch(`/p/${id}/blank.json`).then((r) => r.json()).catch(() => ({}));
  state.height = meta.targetHeight;
  state.hasColour = !!meta.hasColour;
  state.dims = meta.dimensions || null;
  // How closely the mesh matches the photograph. Worth showing because it is the only
  // honest signal that a reconstruction came out badly, and a bad one is better caught
  // here than after somebody has spent ten minutes painting it.
  if (typeof meta.fitScore === 'number') {
    const pct = Math.round(meta.fitScore * 100);
    const el = $('#fit');
    el.hidden = false;
    el.textContent = `Silhouette match ${pct}%`;
    el.className = 'chip ' + (pct >= 80 ? 'ok' : pct >= 65 ? '' : 'bad');
    el.title = pct >= 80
      ? 'The mesh closely follows the photograph.'
      : 'The reconstruction is a plausible version of the object rather than an exact one. '
        + 'More views usually raise this.';
  } else {
    $('#fit').hidden = true;
  }
  // Photographs the projection refused are the one signal that a reconstruction is
  // wrong from a side its own photo cannot show. A build that fits its own photo at
  // 90% can still have a tail turned into a plate; the other views find that.
  const refused = meta.refused || [];
  const warn = $('#warn');
  if (refused.length) {
    warn.hidden = false;
    warn.textContent = `${refused.length} of ${refused.length + (meta.projected || []).length} photographs did not fit`;
    warn.title = refused.map((r) => `${r.side}: ${r.reason}${r.fit ? ` (${Math.round(r.fit * 100)}%)` : ''}`).join('\n')
      + '\nEither the pose differs between shots, or the model is wrong from that side. Retake, or try another photograph as the front.';
  } else {
    warn.hidden = true;
  }
  if (meta.creaseDegrees) {
    $('#crease').value = meta.creaseDegrees;
    $('#crease-out').textContent = `${meta.creaseDegrees}°`;
  }

  const bytes = Uint8Array.from(atob(regions.perTriangle), (c) => c.charCodeAt(0));
  const perTriangle = new Uint16Array(bytes.buffer);

  let mesh = null;
  gltf.scene.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
  const geo = mesh.geometry;
  const idx = geo.index.array;

  // Sort the triangles by region so each region is one contiguous run, then declare
  // a draw group per run. This is the whole trick: after this, painting a region is
  // an array assignment.
  const order = Array.from({ length: perTriangle.length }, (_, i) => i)
    .sort((a, b) => perTriangle[a] - perTriangle[b]);
  const sorted = new Uint32Array(idx.length);
  const ranges = new Map();
  order.forEach((t, n) => {
    sorted[n * 3] = idx[t * 3];
    sorted[n * 3 + 1] = idx[t * 3 + 1];
    sorted[n * 3 + 2] = idx[t * 3 + 2];
    const r = perTriangle[t];
    const e = ranges.get(r);
    if (e) e.count += 3; else ranges.set(r, { start: n * 3, count: 3 });
  });
  geo.setIndex(new THREE.BufferAttribute(sorted, 1));
  geo.clearGroups();
  geo.setAttribute('uv2', geo.getAttribute('uv'));   // aoMap reads uv2

  // A coloured blank also ships a detail normal map on its cube atlas, and the atlas
  // is unwelded (a vertex on a tile seam needs a different UV in each tile), so it
  // cannot sit on the welded blank. Unweld here: three fresh corners per triangle,
  // in the sorted order so the region groups still hold, plus the atlas UVs as uv1.
  const debug = new URLSearchParams(location.search);
  state.atlasUV = false;
  state.hasDetail = !!meta.hasDetail && !debug.has('nodetail') && !softwareGL;
  state.hasGlow = !!meta.hasGlow && !softwareGL;
  state.glowShare = meta.glowShare || 0;
  if (meta.hasColour && !debug.has('noatlas') && !softwareGL) {
    const atlas = new Float32Array(await (await fetch(`/p/${id}/atlas.bin`)).arrayBuffer());
    const T = sorted.length / 3;
    const copy = (name, comps) => {
      const src = geo.getAttribute(name);
      if (!src) return null;
      const out = new Float32Array(T * 3 * comps);
      for (let i = 0; i < sorted.length; i++) {
        for (let c = 0; c < comps; c++) out[i * comps + c] = src.array[sorted[i] * comps + c];
      }
      return new THREE.BufferAttribute(out, comps);
    };
    const uv1 = new Float32Array(T * 6);
    order.forEach((t, n) => { for (let k = 0; k < 6; k++) uv1[n * 6 + k] = atlas[t * 6 + k]; });
    const pos = copy('position', 3), nrm = copy('normal', 3), col = copy('color', 3), uv = copy('uv', 2);
    geo.setAttribute('position', pos);
    if (nrm) geo.setAttribute('normal', nrm);
    if (col) geo.setAttribute('color', col);
    if (uv) { geo.setAttribute('uv', uv); geo.setAttribute('uv2', uv); }
    geo.setAttribute('uv1', new THREE.BufferAttribute(uv1, 2));
    state.atlasUV = true;
    const ident = new Uint32Array(sorted.length);
    for (let i = 0; i < ident.length; i++) ident[i] = i;
    geo.setIndex(new THREE.BufferAttribute(ident, 1));
    geo.deleteAttribute('tangent');
  }
  state.groupOfFace = [];
  for (const r of state.regions) {
    const range = ranges.get(r.id);
    if (!range) continue;
    geo.addGroup(range.start, range.count, r.id);
    for (let i = 0; i < range.count / 3; i++) state.groupOfFace.push(r.id);
  }

  state.meshMats = state.regions.map((r) => materialFor(look.coats[r.id]));
  mesh.material = state.meshMats;
  mesh.geometry.computeBoundingBox();

  if (state.mesh) scene.remove(state.mesh);
  state.mesh = gltf.scene;
  scene.add(gltf.scene);

  // Frame the object by its own size. The camera used to sit where it suits a
  // chair, and an 18 cm figurine at that distance was a speck in the middle of the
  // stage. Same three-quarter angle, distance from the bounding sphere.
  const box = mesh.geometry.boundingBox;
  const size = new THREE.Vector3(); box.getSize(size);
  const radius = size.length() / 2;
  const dist = radius / Math.sin((camera.fov * Math.PI) / 360) * 1.15;
  controls.target.set(0, size.y * 0.45, 0);
  camera.position.set(0, size.y * 0.45, 0).add(new THREE.Vector3(0.62, 0.46, 0.76).normalize().multiplyScalar(dist));
  camera.near = dist / 100; camera.far = dist * 20; camera.updateProjectionMatrix();
  controls.minDistance = dist * 0.15; controls.maxDistance = dist * 4;
  ground.scale.setScalar(Math.max(0.6, radius * 2.5));
  // Unhide BEFORE measuring. resize() reads the canvas parent's box, and a hidden
  // element has no box, so measuring first gives a 0x0 canvas and an empty viewport.
  $('#stage-studio').hidden = false;
  $('#stage-start').hidden = true;
  resize();
  renderMaterials();
  renderRegions();
  select(0);
  $('#ar-link').href = `/ar/${id}`;

}

/* --------------------------------------------------------------- selection */
const ray = new THREE.Raycaster();
canvas.addEventListener('pointerdown', (e) => { canvas._px = e.clientX; canvas._py = e.clientY; });
canvas.addEventListener('pointerup', (e) => {
  // a drag is an orbit, not a click: only treat it as a pick if the pointer stayed put
  if (Math.hypot(e.clientX - canvas._px, e.clientY - canvas._py) > 4) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(
    ((e.clientX - r.left) / r.width) * 2 - 1,
    -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObject(state.mesh, true)[0];
  if (hit && hit.face) select(state.groupOfFace[hit.faceIndex] ?? 0);
});

function select(regionId) {
  state.selected = regionId;
  document.querySelectorAll('.region').forEach((el) => {
    el.classList.toggle('on', Number(el.dataset.id) === regionId);
  });
  syncControls();
}

/* ------------------------------------------------------------------- panel */
function renderRegions() {
  $('#regions').innerHTML = state.regions.map((r) => `
    <button class="region" data-id="${r.id}">
      <i style="background:${state.look.coats[r.id].tint}"></i>
      <span>${r.label}</span>
      <em>${(r.share * 100).toFixed(0)}%</em>
    </button>`).join('');
  document.querySelectorAll('.region').forEach((el) => {
    el.onclick = () => select(Number(el.dataset.id));
  });
}

function renderMaterials() {
  // the photo colour belongs to a specific reconstruction, so it is only offered on a
  // blank that carried colour; the Hunyuan3D path never does
  const list = state.catalogue.filter((m) => m.kind !== 'photo' || state.hasColour);
  $('#materials').innerHTML = list.map((m) => `
    <button class="mat${m.custom ? ' is-custom' : ''}" data-id="${m.id}" title="${m.note}"
      ${m.custom ? `style="background:${m.native}"` : ''}>
      ${m.kind === 'photo'
        ? `<img src="/p/${state.id}/photo.jpg" alt="">`
        : (m.swatch ? `<img src="${m.swatch}?v=${state.matVersion || 0}" alt="">` : '<i class="plain"></i>')}
      <span>${m.name}</span>
      ${m.custom ? '<button class="kill" title="Delete this cloth">x</button>' : ''}
    </button>`).join('');

  document.querySelectorAll('.mat .kill').forEach((el) => {
    el.onclick = async (e) => {
      e.stopPropagation();
      const id = el.closest('.mat').dataset.id;
      // anything wearing it falls back to satin, so the Look stays valid
      state.look.coats.forEach((c, i) => {
        if (c.materialId === id) state.look.coats[i] = { ...c, materialId: 'satin' };
      });
      const r = await fetch(`/api/materials/${id}`, { method: 'DELETE' });
      state.catalogue = (await r.json()).materials;
      renderMaterials();
      patch({});
    };
  });

  document.querySelectorAll('.mat').forEach((el) => {
    el.onclick = () => {
      const def = state.catalogue.find((m) => m.id === el.dataset.id);
      const coat = state.look.coats[state.selected];
      // Picking a fabric brings its own dye with it, unless the person has already
      // chosen a colour they care about. Landing on a grey version of the cloth
      // would make every fabric look identical in the picker.
      const wasNative = state.catalogue.some((m) => m.native
        && m.native.toLowerCase() === (coat?.tint || '').toLowerCase());
      const patchObj = { materialId: def.id };
      if (def.native && (wasNative || !coat || coat.tint === '#b9bed6')) patchObj.tint = def.native;
      // On a coloured blank a repaint starts from the part's own photographed colour,
      // so "matte" on a walnut leg begins as walnut, not as the catalogue's grey.
      const own = state.regions[state.selected]?.photoColour;
      const untouched = !coat || coat.tint === '#b9bed6' || wasNative || coat.materialId === 'photo';
      if (own && def.kind !== 'photo' && !def.custom && untouched) patchObj.tint = own;
      patch(patchObj);
    };
  });
}

function syncControls() {
  const c = state.look.coats[state.selected];
  if (!c) return;
  $('#sel-name').textContent = state.regions[state.selected]?.label || 'none';
  $('#tint').value = c.tint;
  $('#tile').value = c.tileScale;
  $('#tile-out').textContent = `${c.tileScale.toFixed(1)}×`;
  $('#rough').value = c.roughness ?? '';
  $('#metal').value = c.metallic ?? '';
  document.querySelectorAll('.mat').forEach((el) => {
    el.classList.toggle('on', el.dataset.id === c.materialId);
  });
  const def = state.catalogue.find((m) => m.id === c.materialId);
  $('#tile-row').hidden = def?.kind !== 'textured';
  // only the photo material can glow, and only when the photograph had a glow in it
  $('#glow-row').hidden = !(def?.kind === 'photo' && state.glowShare > 0);
  $('#glow').checked = c.glow !== false;
}

/** Apply a change to the selected region, or to everything if "all" is ticked. */
function patch(p) {
  const all = $('#apply-all').checked;
  const ids = all ? state.regions.map((r) => r.id) : [state.selected];
  for (const id of ids) {
    state.look.coats[id] = { ...state.look.coats[id], ...p };
    state.meshMats[id]?.dispose?.();
    state.meshMats[id] = materialFor(state.look.coats[id]);
  }
  state.mesh.traverse((o) => { if (o.isMesh) o.material = state.meshMats; });
  state.dirty = true;
  renderRegions();
  select(state.selected);
  $('#export').textContent = 'Build the AR model';
  $('#export').classList.add('ready');
}

$('#tint').oninput = (e) => patch({ tint: e.target.value });
$('#tile').oninput = (e) => patch({ tileScale: Number(e.target.value) });
$('#glow').onchange = (e) => patch({ glow: e.target.checked });
$('#rough').oninput = (e) => patch({ roughness: e.target.value === '' ? null : Number(e.target.value) });
$('#metal').oninput = (e) => patch({ metallic: e.target.value === '' ? null : Number(e.target.value) });

/* --------------------------------------------------------- the client page */
$('#c-build').onclick = async () => {
  const btn = $('#c-build'), note = $('#c-note');
  if (!state.id) return;
  btn.disabled = true;
  note.textContent = 'Building the page…';
  try {
    const r = await fetch(`/api/projects/${state.id}/dropin`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name: $('#c-name').value, brand: $('#c-brand').value, url: $('#c-url').value }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'failed');
    note.textContent = `${(j.bytes / 1e6).toFixed(1)} MB. Unzip it and run: npx vercel --prod`;
    const a = document.createElement('a');
    a.href = j.zip; a.download = 'client-page.zip';
    document.body.appendChild(a); a.click(); a.remove();
  } catch (e) {
    note.textContent = `Could not build it: ${e.message}`;
  }
  btn.disabled = false;
};

/* ------------------------------------------------------------- cloth maker */
for (const [id, out, fmt] of [['m-threads', 'm-threads-out', (v) => v],
  ['m-rough', 'm-rough-out', (v) => Number(v).toFixed(2)],
  ['m-normal', 'm-normal-out', (v) => v]]) {
  $(`#${id}`).oninput = (e) => { $(`#${out}`).textContent = fmt(e.target.value); };
}
// the quality preset only drives the shape engine's grid; TripoSplat has its own
$('#engine').onchange = (e) => { $('#quality-row').hidden = e.target.value === 'colour'; };

$('#new-mat').onclick = () => {
  const box = $('#mat-maker');
  box.hidden = !box.hidden;
  $('#new-mat').textContent = box.hidden ? 'Invent a cloth' : 'Never mind';
};
$('#m-make').onclick = async () => {
  const btn = $('#m-make');
  btn.disabled = true;
  btn.textContent = 'Weaving…';
  try {
    const r = await fetch('/api/materials', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        name: $('#m-name').value || 'New cloth',
        weave: $('#m-weave').value,
        native: $('#m-native').value,
        threads: Number($('#m-threads').value),
        roughness: Number($('#m-rough').value),
        normalStrength: Number($('#m-normal').value),
      }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    state.catalogue = j.materials;
    // bump the cache-buster so a regenerated map of the same name is not served stale
    state.matVersion = (state.matVersion || 0) + 1;
    renderMaterials();
    $('#mat-maker').hidden = true;
    $('#new-mat').textContent = 'Invent a cloth';
    if (state.look) patch({ materialId: j.id, tint: $('#m-native').value });
  } catch (e) {
    $('#m-note').textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Weave it';
  }
};

/* ------------------------------------------------------------------ export */
$('#export').onclick = async () => {
  const btn = $('#export');
  btn.disabled = true;
  btn.textContent = 'Baking glTF and USDZ…';
  try {
    const r = await fetch(`/api/projects/${state.id}/bake`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(state.look),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    btn.textContent = 'Built. Open on your phone →';
    btn.classList.remove('ready');
    $('#export-note').textContent =
      `${j.materials} material${j.materials === 1 ? '' : 's'}, ${(j.glbBytes / 1e6).toFixed(1)} MB glTF`
      + (j.usdz ? ' and a validated USDZ.' : `. USDZ failed: ${j.usdzNote}`);
    $('#ar-row').hidden = false;
  } catch (e) {
    btn.textContent = 'Build failed';
    $('#export-note').textContent = e.message;
  } finally {
    btn.disabled = false;
  }
};

/* ------------------------------------------------------------------- build */
/* ------------------------------------------------------------------- build */
const views = {};

document.querySelectorAll('.view').forEach((label) => {
  const input = label.querySelector('input');
  input.onchange = async () => {
    const file = input.files[0];
    if (!file) return;
    views[label.dataset.side] = await new Promise((res) => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.readAsDataURL(file);
    });
    // show the photo in the tile so it is obvious which side is which
    label.style.backgroundImage = `url(${views[label.dataset.side]})`;
    label.classList.add('has');
    label.querySelector('em').textContent = 'loaded';
    $('#build').disabled = !views.front || !state.gpu;
  };
});

$('#build').onclick = async () => {
  $('#drop').hidden = true;
  $('#progress').hidden = false;
  const note = (t) => { $('#log').textContent = t; };

  const dims = ['width', 'depth']
    .map((k) => ($(`#${k}`).value > 0 ? `&${k}=${$(`#${k}`).value}` : ''))
    .join('');
  const res = await fetch(`/api/build?height=${$('#height').value}${dims}&quality=${$('#quality').value}&engine=${$('#engine').value}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ views }),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const events = buf.split('\n\n');
    buf = events.pop();
    for (const block of events) {
      const ev = /event: (\w+)/.exec(block)?.[1];
      const data = JSON.parse(/data: (.*)/.exec(block)?.[1] || '{}');
      if (ev === 'progress') note(data.detail);
      if (ev === 'failed') { note(`Failed: ${data.error}`); $('#progress').classList.add('bad'); }
      if (ev === 'done') { await load(data.id); await refreshProjects(); }
    }
  }
};

/* ----------------------------------------------------------------- re-split */
$('#crease').oninput = (e) => { $('#crease-out').textContent = `${e.target.value}°`; };
$('#resplit').onclick = async () => {
  const btn = $('#resplit');
  btn.disabled = true;
  const note = $('#resplit-note');
  const res = await fetch(`/api/projects/${state.id}/refine`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      targetHeight: state.height || 0.78,
      // re-send the blank's own dimensions so a re-split keeps them: under exact
      // scaling the stored dimensions ARE the targets, and for a height-only build
      // they equal what height-only scaling produces, so this is always a no-op or
      // a preservation, never a change
      targetWidth: state.dims?.width || null,
      targetDepth: state.dims?.depth || null,
      creaseDegrees: Number($('#crease').value),
    }),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const blocks = buf.split('\n\n'); buf = blocks.pop();
    for (const b of blocks) {
      const ev = /event: (\w+)/.exec(b)?.[1];
      const data = JSON.parse(/data: (.*)/.exec(b)?.[1] || '{}');
      if (ev === 'progress') note.textContent = data.detail;
      if (ev === 'failed') note.textContent = `Failed: ${data.error}`;
      if (ev === 'done') { await load(state.id); note.textContent = `${data.regions} parts.`; }
    }
  }
  btn.disabled = false;
};

async function refreshProjects() {
  const list = await (await fetch('/api/projects')).json();
  $('#recent').innerHTML = list.slice(0, 8).map((p) => `
    <button class="recent" data-id="${p.id}">
      ${p.regions} parts &middot; ${(p.dimensions.height * 100).toFixed(0)} cm
      <em>${new Date(p.createdAt).toLocaleString()}</em>
    </button>`).join('') || '<p class="none">Nothing built yet.</p>';
  document.querySelectorAll('.recent').forEach((el) => {
    el.onclick = () => load(el.dataset.id);
  });
}

/* -------------------------------------------------------------------- boot */
const status = await (await fetch('/api/status')).json();
state.catalogue = status.materials;
state.matVersion = 0;
state.gpu = status.gpu;
if (!status.colour) {
  // no colour weights on this machine: offer the shape engine only, and say why
  const sel = $('#engine');
  sel.value = 'shape';
  [...sel.options].find((o) => o.value === 'colour').disabled = true;
  sel.title = status.colourReason || 'the colour models are not installed';
}
$('#quality-row').hidden = $('#engine').value === 'colour';
renderMaterials();
$('#gpu').textContent = status.gpu ? 'GPU ready' : status.reason;
$('#gpu').className = status.gpu ? 'ok' : 'bad';
// the build button stays disabled until there is a front view AND a GPU to use it
if (!status.gpu) {
  $('#build').disabled = true;
  $('#build').title = status.reason;
}
await refreshProjects();
resize();
