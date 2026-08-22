# Domain model

The words below mean exactly one thing each in this codebase. Most of the confusion
available in this problem space comes from using "model", "texture" and "part" loosely,
so they are pinned here.

## Nouns

**Source**: the input photograph. One image, one product, plain background if possible.

**Reconstruction** (or **raw**): what the engine hands back, a dense, untextured,
arbitrarily scaled mesh with no UVs, sitting wherever the voxel grid put it, usually
with the studio floor rebuilt as a slab underneath. Half a million triangles. Not
shippable, and never shown to anyone.

**Blank**: the reusable product asset. Floor cut, simplified to a triangle budget,
upright with its feet on y=0 at true size, box-projected UVs, segmented into regions.
**A Blank has no material.** It is built once and never edited. One Blank can wear many
Looks without being rebuilt, which is why the expensive GPU step is paid for once.

**Region**: one contiguous group of triangles the segmentation found, and the unit a
person selects and paints. Regions are discovered by flooding across shared edges and
refusing to cross a crease, *not* by connected components: a reconstruction is usually
one watertight shell, so component labelling would return exactly one region and "select
a part" would mean "select everything".

**Material**: an entry in the catalogue. Either `plain` (paint: no maps, the colour is
entirely the Look's tint) or `textured` (cloth: carries basecolor, normal and ORM maps,
and the tint multiplies over them).

**Coat**: what one Region is wearing: a material id, a tint, a tile scale, and optional
roughness/metallic overrides.

**Look**: the design. One Coat per Region, and nothing else. A Look holds no geometry,
no file paths and no texture bytes, only ids, so it is small, diffable, and survives the
Blank being rebuilt. **The Look is the only mutable thing in the app.**

**Bake**: turning (Blank, Look) into shippable files. Pure with respect to its inputs.

**Export target**: glTF for the web and Android's Scene Viewer, USDZ for iOS AR Quick
Look. Both come out of the same bake so they cannot disagree about the product.

## The state model, in one line

`Blank` is immutable. `Look` is the document. `Bake` is a function of the two.

That is what lets the studio keep an undo stack for free, lets a Look be saved as a few
hundred bytes of JSON, and lets the baker be tested without a browser or a GPU.

## Seams worth knowing about

- **`lib/recon.mjs`** hides ComfyUI entirely. Nothing outside it knows what a
  `Hunyuan3Dv2Conditioning` node is. This is the seam a fixture would sit at to test the
  rest of the app without a GPU.
- **`lib/blank.mjs`** hides the whole difference between marching cubes output and a
  product asset. One function, four numbers.
- **`lib/bake.mjs`** hides both output formats. Callers ask for files, not for a USD
  stage.
- **`server.mjs`** is deliberately shallow. Anything in there that looks like domain
  logic is in the wrong file.

## Decisions that are load-bearing

- **Triangles are sorted by region at load in the browser**, so each region is a
  contiguous draw group with its own material slot. Painting is then an array
  assignment, not a geometry rebuild, which is what keeps the colour picker live at
  60,000 triangles.
- **Regions that wear identical Coats are merged before export.** Twenty-four regions in
  four colourways bakes to four materials, not twenty-four. Both formats pay per
  material and AR Quick Look is noticeably slower with a long list.
- **Groups are split into separate primitives rather than sharing one with offsets**,
  because that is the only portable way to give two parts different texture scales. USD
  has no equivalent of `KHR_texture_transform` that AR Quick Look honours reliably.
- **The transform is baked into the vertices**, not left on the node. USD and glTF
  disagree about how node transforms compose often enough to be worth sidestepping.
- **`ar-scale` is fixed.** The question a shopper is asking is whether it fits, and a
  model they can pinch-shrink cannot answer that.
- **The colour engine's mesh is rebuilt, not repaired.** TripoSplat's mesh is a double
  wall (both sides of the gaussian layer, pinched together at hundreds of points), and
  `lib/carve.mjs` replaces it with a depth-carved visual hull before the budget is
  spent. Anything that edits that mesh face by face is working on the wrong object;
  RESEARCH.md section 0c lists six such fixes that were tried and measured useless.

## Provenance

`lib/comfy.mjs`, the floor cut and the box projection in `lib/blank.mjs`, and the USDZ
approach in `tools/build_usdz.py` were ported from `~/product-viz/tools/` on 2026-08-20,
where they were written and proven. **Those copies and these are now independent.** A fix
in one will not reach the other. The segmentation, the Look model, the multi-material
bake and the studio are new here.
