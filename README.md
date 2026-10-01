# photo-to-ar

One product photograph in. A textured 3D model you can stand in your own room, out.

The reconstruction runs on this machine's GPU, so nothing leaves the building and there
is no per-model licence. Everything after it runs in the browser.

```bash
npm start          # http://localhost:5140
node tools/test.mjs   # 44 end-to-end checks
```

ComfyUI must be running on `:8188`. The shape engine needs
`hunyuan3d-dit-v2.safetensors`; the colour engine needs the TripoSplat stack
(`triposplat_fp16`, `triposplat_vae_decoder_fp16`, `flux2-vae`, `dino_v3_vit_h`) plus a
background-removal model. The app checks both and greys out whichever it cannot run,
rather than failing mid-job.

**Do not launch ComfyUI with `--vram-headroom`.** It reserves that many gigabytes
permanently, and a leftover `--vram-headroom 6` on a 10 GB card leaves about 4 GB
usable, which is below what the colour engine needs.

```bash
cd ~/asset-forge/ComfyUI_windows_portable
./python_embeded/python.exe -s ComfyUI/main.py --windows-standalone-build
```

## What it does

1. **Upload one to four photographs** (front required; left, back and right optional) and
   say how tall the real thing is. Height is the number that makes AR honest: a chair at
   the wrong scale is a toy. Width and depth are optional; with all three the AR
   bounding box matches the real product exactly.

   With the colour engine the extra photographs are also painted onto the model:
   each is aligned by silhouette and projected where the surface faces its camera,
   so the flanks and back carry the real product instead of a guess (on the Shin
   Godzilla figure, three extra views painted 150,000 texels the front could not see).
   A photograph whose pose does not match the model is refused rather than smeared.

   Measured, and against this project's own earlier claim: on a roughly symmetric
   product, one view is enough for the shape. Reconstructing from four consistent views of a known
   mesh scored 0.880 mean silhouette IoU against it across all four sides; one view
   scored 0.875, including 0.872 on the back it never saw. The model's prior invents a
   symmetric back correctly. Extra views are insurance for asymmetric pieces, not a
   general quality lever.
2. **Reconstruct**, with one of two engines.

   Every view is first matted with the segmentation model (BiRefNet), not a colour
   flood fill: on a studio gradient the flood fill kept the caption strip, and the
   shape engine built it as a bar through the figure's head.

   **From the photo** (default) runs TripoSplat: it returns the product's *own colour*,
   recovering cream boucle on a shell and walnut on the legs as different materials.
   About 40 seconds, 6.0 to 6.4 GB of the card's 10. It does its own background removal
   with BiRefNet, which is a trained segmentation model rather than a colour flood fill,
   so it survives a product photographed in a room.

   **Shape only** runs Hunyuan3D 2.0, which returns geometry and nothing else, for when
   you want to dress the thing in the fabric library instead. About twenty seconds of
   sampling, plus a cold-start weight load.

   **Shape from every photo, colour from the front** (hybrid) runs both: Hunyuan3D on
   all the views for the shape, TripoSplat on the front for the colour, then registers
   the colour mesh onto the shape mesh by silhouette and transfers the colour vertex
   by vertex (`lib/hybrid.mjs`). It is the engine for anything whose back matters: on
   the Shin Godzilla figure the colour engine alone turned the dorsal plates into a
   row of knobs, because its one photograph never saw them. About ten minutes. A view
   can be marked shape-only (a photograph of the same mould in another colourway) so
   it guides the shape without painting its colour on; a view under any other name
   (a close-up) is colour-only.

   **Shape from every photo, colour from the front** runs both: Hunyuan3D with all the
   views for the shape, TripoSplat on the front for colour, then registers the two
   (a yaw search on silhouettes; the models' canonical frames differ by about a
   quarter turn) and transfers the colour by nearest vertex (`lib/hybrid.mjs`). On the
   Shin Godzilla figure this is the difference between a row of rounded knobs and
   the real jagged plates down the back, because one photograph shows plates only as
   an outline. It costs about ten minutes, mostly loading a second model onto the
   same card, and the shape engine's proportions fit the front photograph less
   closely than the splat's (76% against 87% on the figure). Best with three or more
   full-figure photographs in the same pose; a close-up among them makes it bend
   things. Detached slabs a multi-view reconstruction leaves floating (a caption
   strip from one photograph) are dropped by distance to the main body.
3. **Finish.** Cut the reconstructed studio floor, rebuild the outer skin when the
   colour engine made it (its mesh is a double wall; see `lib/carve.mjs`), simplify
   half a million triangles to sixty thousand, stand it upright at true size,
   box-project UVs, and segment it into paintable regions.
4. **Paint**, or leave it as photographed. A product that emits light (a figure's
   glow, an LED) is detected from its colour and shipped as emission, with a
   per-part switch. On a coloured reconstruction the first
   material in the picker is "From the photo", which is the product's real colour baked
   into a texture, plus a relief map baked from the dense reconstruction so the
   sixty-thousand-triangle model shades like the half-million-triangle one (a
   figurine's scales and fin edges survive that way). Everything else is a repaint.

   Click any surface on the model to select that part, then give it a colour,
   a material and a texture scale. Roughness and metallic are there when you want them.
   "Invent a cloth" generates a whole new material from a weave structure, a thread
   count, a roughness and a relief, in about a second, and it joins the picker.
5. **Export.** A glTF and a matching USDZ, from one bake, so Android and iOS cannot
   disagree about the product.
6. **Hand it to someone.** `node tools/dropin.mjs <projectId> --out <dir> --name "Boucle
   Lounge Chair" --brand "Workflo" --url https://where-it-will-live` writes a folder any
   static host serves: the product in a viewer, the AR button, the true size, a QR code
   for the person reading on a desktop, and the USDZ content type a host otherwise gets
   wrong. `--deploy` runs Vercel for you. The studio does the same behind "Make a page
   for a client" and hands back a zip.
7. **Open it on a phone.** The AR page shows a QR code for this machine's secure
   address, and serves a Draco-compressed twin of the glTF (about 40% of the size)
   with a local decoder, falling back to the plain file. The server prints two
   network addresses on startup. The
   plain `http://` one works for iPhone (AR Quick Look does not care). **Android needs the
   `https://` one on the next port**: WebXR requires a secure context and Scene Viewer
   requires an https model URL, so over plain http the AR button on Android does nothing
   and says nothing. The certificate is self-signed; accept the warning once. Generate
   it with the command in `certs/san.cnf`'s neighbour comment in `server.mjs`, and put
   your own LAN address in the SAN list.

## The parts that were actually hard

**The floor.** A single-image reconstruction cannot tell the product from the set it was
shot on, so the seamless and its contact shadow come back as a slab of geometry under
the piece. You cannot find it by vertex count, because simplification collapses a flat
plate down to almost nothing. You can find it by width: the slab spans the whole frame
while the legs standing on it span a few centimetres. Find the lowest band that is
anomalously wider than the band above it, and cut.

**Per-part selection on a fused mesh.** The obvious approach, connected-component
labelling, is useless here: a reconstructed chair is one watertight shell, so every
triangle is one component. What separates a seat from a back is curvature, and what
separates a walnut leg from a boucle shell is neither: it is material. So the flood
refuses to cross two things, a crease and a change of material, and the second one
needs a global answer. An edge-local colour test walks from cream to walnut along a
gradual path, and on a photo-derived texture there is always a gradual path: measured
on a shoe, it found five regions at one threshold and one at the next. Clustering the
whole surface first (area-weighted k-means on an opponent-space colour, clusters closer
than the threshold merged) gives every triangle a label no path can smear. The chair
then splits into a cream shell and four walnut legs at 38 degrees, where it used to need
the crease angle walked all the way down to 8 to find anything at all.

The crease half:

The threshold cannot be a constant, and finding that out was instructive. Tuned against
plain marching cubes it sat at 38 degrees, because the stair-stepping supplied creases
everywhere. Switching to a smoother isosurface removed those artifacts, which is the
whole point of switching, and the same 38 degrees then returned **one region covering the
entire chair**. Most of what the original setting was finding was faceting, not
furniture. So the angle now walks down (38, 26, 18, 12, 8) until the split is actually
useful: enough parts, none of them swallowing the object.

That flood also leaves a long tail of two-triangle slivers along the creases themselves,
which is why anything under 0.4% of the surface is folded into whichever kept region it
touches. Offering someone 4,000 clickable regions is the same as offering them none.

**Making the two AR formats agree.** They are built from one bake, at metric scale, with
the same material split. The USDZ is served as `model/vnd.usdz+zip`; a host that guesses
`application/octet-stream` leaves the AR button doing nothing, with no error anywhere,
which is a miserable thing to debug on a phone.

## Layout

```
lib/recon.mjs      photograph -> raw glb        (hides ComfyUI entirely)
lib/blank.mjs      raw glb -> Blank             (floor, simplify, scale, UV, segment)
lib/carve.mjs      splat mesh -> outer skin     (depth-carved hull + surface nets)
lib/hybrid.mjs     shape mesh + colour mesh     (register by silhouette, transfer colour)
lib/detail.mjs     raw mesh -> relief + colour  (per-texel from the dense mesh)
lib/project.mjs    photographs -> atlas         (silhouette-aligned projection)
tools/dropin.mjs   a project -> a client page   (static folder, AR, QR, USDZ mime)
lib/hybrid.mjs     shape mesh + colour mesh     (register by silhouette, transfer colour)
web/inspect.html   ?id=&yaw=&flat=1             (dev view of a baked model, relief on/off)
showcase/          the Shin Godzilla showcase   (build_showcase.py -> one self-contained page)
lib/materials.mjs  the catalogue
lib/look.mjs       the document, pure data
lib/bake.mjs       Blank + Look -> glTF + USDZ
tools/build_usdz.py  the iOS half, via OpenUSD
web/               the studio and the AR page
projects/<id>/     source.png, raw.glb, blank.glb, regions.json, look.json, model.*
```

`CONTEXT.md` has the domain model and the decisions that are load-bearing. Read it before
changing anything in `lib/`. `PLAN.md` is the record of the last improvement round, with
what was measured and what was not done. `RESEARCH.md` sections 0 to 0f are the findings
in order.

## Getting colour out of a splat, and into a format iOS will read

TripoSplat's mesh carries per-vertex colour on an unlit material: no UVs, no images. AR
Quick Look reads neither vertex colours nor displayColor, so shipping that straight
would render **plain grey on an iPhone with no error anywhere**. The colour has to
become a real texture, and a texture needs a unique UV layout.

The Blank's usual UVs are box-projected in metres so a weave tiles at a real scale.
Those tile, which is the opposite of unique: many points on the surface share a texel.
So the coloured path gets its own layout, a six-tile cube atlas keyed on the dominant
**signed** axis of each face normal. Signed, not unsigned, because an unsigned box
projection maps the front and the back of an object to the same place, which is the same
overlap in a smaller disguise. Per face, not per vertex, or a triangle whose corners
disagree gets stretched across two tiles.

Two things went wrong there that are worth knowing about, because neither looked like an
error:

- **Barycentric weights belong to the OPPOSITE vertex.** An edge function evaluated at a
  point is proportional to distance from that edge. Rotating the three weights by one
  scatters each triangle's darkest corner into its lightest neighbour, which reads as
  noise rather than as a bug.
- **COLOR_0 is linear; a baseColorTexture is sRGB.** Writing linear values straight into
  the image skips the transfer function and crushes the shadows: a linear 0.036 became a
  near-black texel instead of the 0.20 mid-shadow it stands for. The whole model looked
  speckled until this was fixed.

## The studio's lighting is calibrated, not flattering

Scene Viewer and AR Quick Look light a model with a neutral environment whose
irradiance is about one, so an albedo maps to roughly itself. The studio used to put
about 3.5x that on a lit face (hemisphere 1.5, key 2.4, rim 1.0): a cream chair went
chalk white and a near-black figurine, every vertex under 0.05 luminance, rendered mid
grey. It now uses three.js's procedural RoomEnvironment at 0.9 with two low
directional lights for shape, and the photo material is matte (roughness 0.9, quarter
specular) because the photograph's own highlights are already in its colour.

On a software GL (SwiftShader in headless Edge, which is what `tools/test.mjs` drives)
the studio skips the relief map and the unwelded atlas geometry: a 180k-vertex mesh
with a normal map took that renderer minutes per frame and the harness's screenshot
never returned. A real GPU gets everything.

## Two conventions worth knowing

glTF's texture origin is top-left (v grows down the image); OpenGL's and USD's is
bottom-left. The atlas is written in glTF's; the USDZ builder flips v; the studio sets
`flipY = false` on atlas textures. For a day it was the other way round and every glTF
viewer showed each face the opposite side's colour while the studio looked fine.
`tools/test.mjs` now samples the texture at vertices' UVs and compares with the vertex
colour, which fails under a flip.

The alignment camera's up vector is `forward x right`. `right x forward` points down,
and a silhouette rendered upside down still scores 0.73 against an upright chair.

## What has not been verified

**AR on a physical handset.** Everything up to the file is checked: the USDZ is opened
back up and validated after packaging, the content types are asserted, and both formats
are built from the same bake. But no phone has actually placed one of these in a room.
Until that happens, treat the AR half as "correct by construction" rather than as
working.

The reconstruction is also only as good as the photograph. One view means the back of
the object is invented, and it is invented smoothly. Thin structures, deep concavities
and anything transparent come back wrong.
