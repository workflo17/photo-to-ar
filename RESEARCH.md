# Where this could go next

Updated 2026-08-21 after a seven-lens research fan-out with adversarial verification.
The headline is a REVERSAL: the previous edition of this file concluded that local
texture generation was out of reach on a 10 GB card. **That was wrong**, and the
correction is the first section below.

Two kinds of claim, kept separate: things TESTED on this machine, and things READ. The
hardware still frames everything: one RTX 3080 with 10 GB, and, newly relevant, only
about 10 GB free on the C: drive.

---

## 0. Correction: local photo-derived texture DOES fit, via TripoSplat

**Tested on this machine. 6.0 to 6.4 GB peak of 10.0. About 30 seconds per model.**

TripoSplat ships as core ComfyUI nodes, is MIT, and its whole stack is 3.6 GB of fp16
weights that are now on disk. One photograph in, a coloured mesh out. On the reference
chair it recovered the cream boucle AND, separately, the walnut legs, which is a
per-part material distinction `lib/weave.py` cannot express at all.

Its shape scores **0.729** on this app's own silhouette metric against the Hunyuan3D
blank's 0.725, so it is not a downgrade in geometry either.

Node chain: `LoadImage -> LoadBackgroundRemovalModel(birefnet) -> RemoveBackground ->
TripoSplatPreprocessImage -> CLIPVisionLoader(dino_v3_vit_h) + VAELoader(flux2-vae) ->
TripoSplatConditioning -> UNETLoader(triposplat_fp16) -> KSampler -> 
VAELoader(triposplat_vae_decoder_fp16) -> VAEDecodeTripoSplat -> SplatToMesh -> SaveGLB`.
Every node reports `api_node=False`, so nothing leaves the building.

**The trap, and it is a serious one.** `SplatToMesh` emits vertex colours on a 2.2M
triangle **unlit** mesh: no UVs, no images, no textures. AR Quick Look does not read
vertex colours, so a USDZ built straight from that renders **plain grey on iPhone with
no error anywhere**, the same failure mode as the content-type bug. `lib/bake.mjs` must
rasterise the vertex colours into a baseColor texture through the box-projected UVs the
Blank already carries, and drop the unlit flag. The colour also has studio lighting
baked into it, so it wants de-lighting before it is treated as albedo.

**Why the old conclusion was wrong:** Hunyuan3D 2.1's 21 GB really is out of reach, and
DynamicVRAM cannot rescue it because that 21 GB is activation memory, not weights. The
error was concluding that ALL local texture was therefore impossible, when a different
model with a different memory profile fits comfortably.

Related: the ComfyUI instance this app had been measuring against was launched with
`--vram-headroom 6`, permanently reserving 6 of the card's 10 GB. Every VRAM figure
taken before that was found was measured against a card two fifths its real size.

**Effort:** days, and most of it is the bake, not the generation. **Risk:** the grey-on-iOS
trap above, plus 3.6 GB of disk on a drive with about 10 GB free.

---

## 0b. Three AR bugs, found and FIXED 2026-08-21

All three were in shipped code, all three were silent, and all three now have
regression tests that read the built file back (`tools/usdz_facts.py`, asserted in
`tools/test.mjs`).

1. **The tint never reached iOS.** `build_usdz.py` carried the Look's colour on
   `primvars:displayColor`. RealityKit does not read displayColor or vertex colours at
   all, so the same chair rendered navy on Android and untinted grey on iPhone. On the
   reference project that was 48,684 of 60,000 triangles wearing the wrong colour. The
   tint now rides on the base sampler's `inputs:scale`, which is UsdUVTexture's
   documented multiply and is honoured.
2. **Normal maps were lit wrong on iOS.** UsdPreviewSurface does not unpack an 8-bit
   normal map for you; the spec requires `scale=(2,2,2,1)` and `bias=(-1,-1,-1,0)`.
   Without them a flat texel is read as a normal tilted toward +X+Y, so every textured
   surface in the package was systematically wrong.
3. **No tangent basis anywhere.** `mikktspace` was in package.json and imported by
   nothing. With box-projected UVs, which mirror on opposite faces and seam at every
   axis crossing, three.js, Scene Viewer and RealityKit each derived a different basis,
   so the same fabric caught light three different ways on three surfaces. The bake now
   runs unweld, tangents, weld.

## 0c. The flecks on the colour engine's mesh, found and FIXED 2026-08-21

The coloured reconstruction carried grey flecks across its upper back and arm tops
that survived every plausible fix, and the record of the failures is the useful part.

**What they were not.** Each of these was tested by rebuilding and rendering, and each
left the render unchanged: duplicate faces (8.8% of the mesh, real, removed, no
effect on the flecks); floating fragments under thirty faces (removed, no effect);
dark colour outliers in small clusters (inpainted, no effect, and the same rule later
painted the walnut legs cream); vertices displaced off the surface (lifted, no
effect); short edges collapsed (124 of them, no effect); more smoothing passes (30
instead of 8, softened but still there). A flat render with no colour finally showed
they were geometry, and a probe of the triangles under one fleck showed what kind.

**What they were.** The mesh is a thin double wall. Meshing a layer of flat gaussians
produces both sides of the layer: an outer sheet the camera saw at 292 to 306 mm from
the probe camera, and a ghost sheet at 280 to 289 mm facing the other way, with the
two pinched together through five faces a tenth the usual size. Smoothing, which is
needed for the rest of the surface, drags every pinch into a tunnel whose edge-on
walls render as a fleck. Measured visibility from 128 directions: 57% of the mesh's
area is seen from nowhere.

**What fixed it.** Not repairing the mesh: the ghost sheet is woven into the outer
one at hundreds of points, and a visibility cull with hole filling, tried in full,
left 382 dimples behind. Instead `lib/carve.mjs` rebuilds the outer skin from what
can be seen of it: depth from 64 directions carves a 192-voxel grid, surface nets
mesh the result, and each new vertex takes the nearest original vertex's colour. The
first render was clean. Cost 34 s in the build at 4 mm voxels on a chair.

**Side effects worth knowing.** With a carved skin the normals are real, so the
de-light stage (section 0) started firing at a 7% fit and brightened the legs; its bar
is now 15%. The fleck inpainting built for the old mesh is gone for the same reason.

## 0d. Detail without triangles, and a figurine as the test piece, 2026-08-21

A Shin Godzilla figure (S.H.MonsterArts, official product photographs) was the first
subject with fine detail, and it exposed three things the chair could not.

**Surface nets on a binary grid fold.** The carve's first mesh had 16.5% of its
corners more than 60 degrees off their own face (pinched diagonal contacts and the
staircase), and simplification doubled that to 35%, which rendered as scattered flat
facets in every viewer. Extracting the 0.5 level set of a blurred occupancy (two
passes of a separable 3-tap box) with interpolated edge crossings brought it to 0.5%
before and 0.9% after simplification, and halved the triangle count for free.

**The relief map (`lib/detail.mjs`).** Per texel of the colour atlas, the closest
point on the dense mesh, its interpolated normal, expressed in the coarse triangle's
tangent frame. Three sources were tried. A blend of nearby dense vertex normals left
a self-bake (coarse against itself, which must be flat) at 24 degrees, because the
ball reaches through a tail or a fin; accepting only same-facing vertices got 10;
closest-point-on-triangle got 1.1. The carved hull as the source baked its own voxel
staircase as 20 degrees of speckle; the raw splat mesh is the source now, and its
ghost sheets and tunnel walls are excluded by the same facing test. Measured relief on
the figure: 11 to 21 degrees mean, depending on the photograph.

**Lighting was the "grey Godzilla".** Every vertex under 0.05 luminance, rendered mid
grey, and neither specular nor de-light was the cause: the studio's rig put 3.5x the
irradiance an AR viewer does on a lit face. Calibrated to a neutral environment; see
README. De-light's bar moved to a 15% fit because real normals made a 7% fit apply
and brighten the walnut legs.

**The photograph decides.** Same figure, same pipeline, 71 s each: front view 82%
silhouette match with the stance and the face; side view 70% with the whole tail but a
thinner body; three-quarter view turned the tail into a sheet. Showcase page:
`showcase/godzilla/build_showcase.py`, published as the "Shin Godzilla Scan" artifact.

## 0e. Four improvements, two bugs that hid behind them, 2026-08-21 night

Asked for: colour on the unseen sides, sharper thin parts, a real unwrap, and glow.

**Glow (`glow.jpg`).** Saturated colour 3 to 6x brighter than the body's median, on
under a fifth of the surface, is emission. Ships as glTF emissiveTexture and USD
emissiveColor; a coat flag turns it off per part. 2 to 4% of the Shin Godzilla figure.

**SDF fusion replaced binary carving.** A truncated signed distance fused from the same
64 depth maps, with one three-tap smoothing pass, extracted at its zero level set.
Folded corners 1.6% before simplification; fins and tail ridges survive that the
blurred binary grid softened. Unsmoothed, the fused field folded 13.8%: views
disagree by a wall's thickness wherever the raw mesh is double-walled.

**xatlas, measured and NOT shipped.** On a clean Hunyuan3D mesh: 1,871 charts, 64%
coverage, 36 s. On the carved meshes this app makes: 2,500 to 4,100 charts, 17 to
25% coverage (the cube atlas gets 30 to 36%), 10 to 190 s, and no chart option moved
the count. The parameteriser fails on thin high-curvature parts and falls back to
piecewise charts. The real texture-detail loss was elsewhere: the atlas was
interpolating the coarse mesh's 30k vertex colours. **Colour is now sampled per texel
from the raw mesh** through the relief bake's closest-point lookup (with a fallback to
the coarse colour where nothing is in reach, and a 5x5 median). The eye, the teeth and
the skin folds appeared.

**Photographs painted on (`lib/project.mjs`).** BiRefNet mask, silhouette alignment,
per-texel projection with a depth test, exposure matched to the existing colour, a
refusal when the gains leave 0.5 to 1.9 (a close-up of the back matched the front's
silhouette at 77% and would have painted red fins across the chest). Four views of
the figure: 87 / 78 / 73 / 65% fits, 215k texels painted; the two photographs with
a different tail pose were refused at 32 and 55%.

**Two bugs found under that work, both old.** The alignment silhouette was upside
down (`up = right x forward` points down); a chair's silhouette is nearly symmetric
top to bottom so the scores looked plausible (0.73), a standing figure's were not
(0.28 upright, 0.81 flipped). Fixed: the chair is 95%, the figure 87%. And the colour
atlas was written in OpenGL's v convention while glTF's origin is top-left, so every
glTF viewer showed each face the opposite side's chart, mirrored; the studio and the
USDZ read the other way and hid it. Both now have tests that fail under a flip.

## 0f. The seventeen-item plan, 2026-08-22 early

`PLAN.md` has the item-by-item record. The findings that change how the pipeline
should be thought about:

- **Alignment was slow for a silly reason.** The silhouette rasteriser built a
  closure and two arrays per vertex; removing that took a silhouette from 18 ms to
  6.7 ms, and caching the per-view projection across the placement search took a
  photograph's alignment from 21 s to 1.8 s. Coarse-to-fine searching, the clever
  idea, was a net loss on its own.
- **Blend photographs winner-takes-most.** Three photographs registered a few
  millimetres apart are three slightly different Swooshes, and averaging them at
  facing^2 is a blur. At facing^8 the best-facing photograph wins almost outright,
  and the reconstruction's own colour only speaks where no photograph faced the
  texel above 0.5. This, plus reaching colour into grazing-angle texels, is what
  made the shoe legible.
- **Silhouette carving needs near-perfect placement.** At a 79% fit a three-quarter
  view carved real hip and leg volume off the figure and dropped its own front fit
  from 87% to 78%. Gated at 85% it helps the shoe (four views at 87 to 98%) and
  leaves the figure alone. A silhouette may only subtract when it is placed exactly.
- **Gloss from view disagreement is not measurable here.** Brightness ratios between
  photographs of the same texel: median 2.9 on the figure, 1.5 on the shoe. That is
  registration error, and a roughness map from it would be noise.
- **Glow is relative to white, not to the median.** A pastel shoe "glowed" over 12%
  of itself against its own median; against its white midsole nothing does, and the
  figure keeps its strong glow (1.6%).
- **Thin-sheet share does not catch the sheet-tail failure** (that plate was a
  centimetre thick). The other photographs' refusals are the detector, and a
  single-photograph build has no detector at all.
- **Not done, honestly:** octree voxels near thin features, per-part pose-tolerant
  alignment, and a trained part segmenter (no SAM3 weights on disk, PartField needs
  its own environment).

## 0g. The dorsal plates, and the hybrid engine, 2026-08-22

The owner's note: "spines on godzilla's back is weird". They were: a neat row of
rounded knobs. The raw TripoSplat mesh already had them that way, so no stage of the
pipeline was to blame; a single front photograph shows the plates as an outline and
the model's prior fills in generic spikes.

Hunyuan3D's multi-view checkpoint, fed all four photographs, returns jagged layered
plates down the back and tail, teeth in the mouth, and no colour. `lib/hybrid.mjs`
registers the splat mesh onto it (yaw search scored by silhouette agreement from five
directions; the two frames differ by 276 degrees here) and transfers the splat's
vertex colour by nearest neighbour (7 mm away on average at this scale). The finish
then runs as for any coloured raw mesh: dense colour and relief from the raw mesh,
photographs projected on top. The back close-up aligned at 87% and painted the glow
between the plates, once photographs placed that well were trusted over the base
colour instead of being refused for disagreeing with a guess.

What it cost, measured: the front fit is 76% against the splat's 87% (the shape
engine bent the tail forward, having to reconcile a close-up "back" view with three
full-figure ones), the chest took misplaced red from three-quarter photographs at 70
to 74% fits, and the build is about ten minutes. The caption strip in the close-up
came back as a slab floating above the head; a detached-piece pass now drops pieces
whose vertices are all further than 2.5% of the height from the main body (a box
test failed, because the raised tail tip put the slab inside the box). ComfyUI's
liveness probe also needed retries: it stalls for seconds while swapping models and
a single four-second probe reported it dead mid-stage.

So the showcase keeps the single-photograph colour build as the hero and shows the
hybrid's back beside it. Neither is strictly better; the hybrid is the one to reach
for when a product's defining detail is on a side the front cannot show, and it
wants full-figure photographs in one pose.

---

The hardware is the frame for all of it: a single **RTX 3080 with 10 GB**. That number
decides more than any software choice.

---

## The ceiling, revised

Hunyuan3D 2.1's texture stage really is out of reach: 21 GB for texture, 29 GB for the
full pipeline, and that 21 GB is **activation** memory, so DynamicVRAM and aimdo cannot
page around it. TRELLIS at 16 GB is also above the line. MV-Adapter wants 14 GB,
Hunyuan3D-2GP 24.5 GB.

What was wrong before was the inference drawn from that: not "no local texture", only
"not from those models". TripoSplat fits in 6 GB and is covered in section 0.

For **geometry** the position stands. The open image-to-3D field is flat: swapping the
generic reconstruction model buys roughly nothing, and the app's measured weakness is
subject-dependent rather than model-dependent. On the reference chair IoU is 0.725; on a
different photograph the same pipeline scored 0.43, and that spread dwarfs any model
choice. (Both numbers were taken with the upside-down silhouette of section 0e; the
chair re-scores at 0.95 with it fixed.)

## 1. Take more than one photograph

**Tested on this machine. Works. No download, no new model, no cost.**

`Hunyuan3Dv2ConditioningMultiView` is already in this ComfyUI and accepts four optional
views: front, left, back, right. I ran the full graph through it against the existing
checkpoint and it was accepted, producing a mesh in 68 seconds.

This matters more than any setting. Single-image reconstruction **invents the back of the
object**, smoothly and confidently, and no octree resolution or isosurface algorithm
fixes an invention. Right now the app asks for one photo while the model on disk accepts
four.

Nothing exotic is required of the user: walk around the chair and take four pictures.
That is the single biggest quality improvement available, and the work is entirely in the
upload UI and the graph, not in any model.

**Effort:** a morning. **Risk:** low, the graph is proven.

---

## 2. Replace the matte with a real segmentation model

**Local, small download.**

**Update: the weights are now on disk.** The TripoSplat download brought
`birefnet.safetensors` (444 MB) with it, so `models/background_removal/` is no longer
empty and both `LoadBackgroundRemovalModel` and `RemoveBackground` run. They executed
inside every TripoSplat test. This is now a two-node graph change, not a download.

`RemoveBackground` and `LoadBackgroundRemovalModel` are both `api_node: False`, so they
run here.

The current matte (`lib/matte.py`) is a Lab flood fill from the frame edge. It is decent
on a studio seamless and it will **fail completely on a real photograph**: a chair on a
rug, in a room, with a skirting board behind it. Since matting now runs before
reconstruction and directly determines whether a floor slab gets built at all, this is
the input the whole pipeline rests on.

`SAM3_Detect` is also installed and local, and would do the same job with prompting.

**Effort:** an afternoon. **Risk:** low.

---

## 3. Camera projection: attempted, blocked, and now SUPERSEDED by section 0

**Built the alignment, measured it, and stopped. The blocker is not the alignment.**

> Superseded: TripoSplat (section 0) gets photo-derived colour without needing to solve
> the camera at all, because the model produces colour directly rather than having a
> photograph projected onto it. The alignment work below still earns its place as the
> app's quality metric, and the reasoning about why projection fails is still the reason
> not to revisit it.

Camera-projection baking needs to know where the photograph's camera was. Hunyuan3D
returns no camera, so `lib/align.mjs` finds one: it rasterises the mesh silhouette from
a sweep of candidate cameras and scores each against the matte's own subject mask by
intersection over union, then hill-climbs scale and offset for the best few. Coarse over
180 orientations plus refinement takes about seven seconds.

On the reference chair the best fit is **IoU 0.725** at yaw 180, pitch 20.

That is not enough to project onto, and the reason matters. It is tempting to blame the
contact shadow, which the photograph's mask contains and the mesh cannot, since the floor
was cut. Measured: the shadow is **5% of the disagreement**, and scoring with it excluded
gives 0.725, exactly the same. The error is not placement and not the shadow. It is
**shape**: 3,867 pixels of mesh with no photograph under them, and 2,657 pixels of
photograph with no mesh.

A single-view reconstruction produces a plausible chair, not that chair. Projecting a
photograph onto a mesh that is the wrong shape slides the texture off the form and drags
backdrop colour onto the silhouette edges, and no better alignment fixes it because the
alignment is already close to the best available.

Two further problems sit behind this one, both unaddressed: the photograph carries baked
studio lighting, so a projected texture is lit twice, once in the pixels and once by the
viewer; and one view covers one side, leaving the back to be mirrored or invented.

**What came out of the attempt anyway**, and is now in the app:

- The silhouette score is reported per build and shown in the studio, because "how well
  does this mesh match the photograph" is the one honest measure of whether a
  reconstruction came out well, and it is better seen before someone paints for ten
  minutes than after.
- Two real bugs, both found by looking at intermediate images rather than at code. See
  the matte notes below.

**What would unblock it:** a more faithful mesh. Multi-view reconstruction is the cheap
route and is already shipped, so the first thing to try is whether four real photographs
lift the score. Beyond that the honest answer is that texture belongs to a model built to
generate it, which on 10 GB means the cloud nodes.

## 3b. Paint the product's own colour onto it, original plan

**Code only. No model, no VRAM.**

Camera-projection baking is old, well-understood technique, not a research problem: give
the mesh UVs from the source camera's point of view, sample the photograph through them,
and bake the result into the real UV layout.

Hunyuan3D reconstructs roughly aligned to the input view, so the side facing the camera
can take its actual colour from the photograph. That is the difference between "a chair
shape wearing synthetic bouclé" and "this chair". The back has no data and would need
mirroring or inpainting, which should be visible in the UI rather than hidden.

This is the most realistic route to real product colour on this hardware, precisely
because it needs no model at all.

**Effort:** two or three days. **Risk:** medium, the alignment is approximate and needs
measuring before promising anything.

---

## 4. Segment parts with a model instead of a heuristic

**Local, real download, most work of anything here.**

The crease-angle ladder is a heuristic and it shows: on the reference chair the largest
part is 83% of the surface, because a tub chair's shell genuinely is one continuous
curve. It cannot tell a seat from a back the way a person does.

**PartField** (NVIDIA, ICCV 2025) predicts part feature fields that cluster into
decompositions, is open source with inference code, and is reported to beat the earlier
open-world methods on both quality and speed. **PartSAM** (ICLR 2026) reports better than
20% IoU over the next best.

Either would replace the heuristic with something that returns seat, back, arms and legs
because it understands objects, not because the mesh happens to fold there.

**Effort:** a week, including finding out whether it fits in 10 GB. **Risk:** medium to
high, and worth reading the VRAM numbers before starting.

---

## 5. Unwrap properly instead of box-projecting

**Code only.**

UVs are box-projected: each vertex picks the axis its normal most points along. It is
cheap, it tiles predictably, and it **smears on every surface facing diagonally**, which
on a rounded chair is most of it. A real unwrap (xatlas is the usual choice and has a
wasm build) would give clean, non-distorted texture space.

The catch is that box projection is what makes texture scale meaningful in metres, so a
switch means rethinking how tile scale is expressed. Worth doing alongside item 3, since
a projected photo texture needs a proper unwrap far more than a tiling weave does.

**Effort:** two days. **Risk:** medium, it touches the one thing currently working well.

---

## 6. Generate material sets instead of shipping four

**Code only, already half-built elsewhere.**

`product-viz/tools/make_fabrics.py` is fully parametric: the entire library is a
four-row table of weave structure, albedo, roughness, normal strength and thread count.
Wiring those to a form turns four fixed fabrics into an open family, and the generator is
already proven.

This is the cheapest item on the list and the most directly aimed at "customisable
textures". It pairs with the colour fix already made, since the dye is now separated from
the weave anyway.

**Effort:** a day. **Risk:** low.

---

## The cloud option, stated plainly

These are already installed and would solve items 3 and 4 outright, today:

| Node | What it does | Cost |
| --- | --- | --- |
| `Tencent3DTextureEditNode` | textures an existing mesh | paid API |
| `Tencent3DPartNode` | part segmentation | paid API |
| `TripoImageToModelNode`, `TripoMultiviewToModelNode`, `TripoTextureNode` | reconstruction and texturing | paid API |
| `Rodin3D_*`, `MeshyTextureNode` | the same again | paid API |

All are `api_node: True`. They would work now, they would look better than anything
local on a 10 GB card, and they would **send the customer's product photograph to a third
party** and put a per-model cost on something that currently costs nothing to run.

That trade is a business decision, not a technical one. It is worth noting that the
privacy claim is currently a selling point of this app, and a hybrid ("local by default,
cloud when you ask for it") keeps the claim honest while making the ceiling optional.

---

## If only three things get done

1. **Multi-view input.** Tested, free, and fixes the invented back, which is the largest
   remaining source of wrongness.
2. **A real background remover.** Cheap, and the current one will fall over on the first
   photograph taken outside a studio.
3. **Parametric materials.** A day's work aimed straight at what was actually asked for.

Items 3, 4 and 5 in the list above are the ambitious ones and should wait until the first
three are in, because each is easier to judge once the input is good.
