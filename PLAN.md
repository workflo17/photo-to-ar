# Improvement plan, 2026-08-21 night, closed 2026-08-22 early

Seventeen items from the post-showcase review. Each is marked by what was measured,
not by what was written. The test suite stands at 40 checks, all passing.

## Phase A: quick wins
- [x] A1 [2]  Colour atlas at 2048; relief and glow at 1024. Dense bake and projection follow the atlas size.
- [x] A2 [15] Alignment 21 s -> 1.8 s per photograph. The cost was a closure per vertex in the silhouette rasteriser (18 ms -> 6.7 ms per silhouette), plus the per-view projection cached across the placement search; the fit stage's result is reused by the projection.
- [x] A3 [1]  Tile assignment by neighbourhood majority plus absorption of charts under six faces: 1,221 charts -> 513, single-face charts 562 -> 111 on the figure.
- [x] A4 [9]  Refused photographs recorded with reasons and shown as a chip in the studio. Thin-sheet share is computed and stored but does NOT detect the sheet-tail failure (that plate was a centimetre thick); the other photographs' fits are the detector.
- [x] A5 [11] Each part's photographed colour (`photoColour` in regions.json) is the starting tint for a repaint.
- [x] A6 [13] Draco twin (`model-lite.glb`, 3.5 MB -> 1.4 MB on the figure) served to the AR page with a local decoder and a fallback to the plain file.
- [x] A7 [14] QR code on the AR page for the https LAN address.

## Phase B: colour quality
- [x] B1 [3]  Unseen texels reached from painted neighbours within each chart, fading to the guess with distance. Texels seen only below facing 0.5 count as unseen, which removed the dark speckle on the shoe's heel.
- [x] B2 [4]  Per-photograph de-lighting: fit accepted on the chair's front (29% explained, light from upper left), refused on all four figure photographs (2 to 7%), which is the right call for a glowing, textured subject.
- [~] B3 [5]  Measured and NOT shipped: brightness ratio between photographs of the same texel has a median of 2.9 on the figure and 1.5 on the shoe. That is registration error, not gloss; a 65 to 97% silhouette fit does not align to the texel. The number stays in the log as a registration diagnostic.
- [x] (found on the way) Winner-takes-most blending (facing^8) and the guess stepping aside wherever any photograph faced a texel above 0.5. Averaging three photographs registered millimetres apart had turned the shoe's Swoosh into a blur.

## Phase C: geometry (closed 2026-08-22, each item measured)
- [~] C1 [6]  Second pass carving with the photographs' silhouettes exists and runs, gated at 85% fit per photograph. At 70% it took real hip and leg volume off the figure and dropped its own front fit from 87% to 78%; at 85% it runs on the shoe (four photographs at 87 to 98%) and not on the figure (best 81%). The hybrid engine does not carve, so it never runs there.
- [~] C2 [8]  Built and switched OFF. Block-sparse refinement (`refine: 2` in carve.mjs): only the 8-voxel blocks that contain the surface get a 2x finer grid fused from the same 64 depth views, extracted by surface nets over the sparse fine grid. Measured on the figure: 15k blocks, 62M fine voxels, a 2.9M-triangle hull in 72 s instead of 34, and after simplification to the 60k-triangle budget the shipped mesh was no sharper and showed more facets. The budget is the limit, and the relief map carries the detail. Available for a subject that warrants a bigger budget.
- [~] C3 [7]  Partly. Per-part exclusion exists: when the whole silhouette fails, parts the photograph's mask covers under 50% at the best view (at most 40% of the surface) are left out, the rest is aligned again, and if it fits, the rest is painted. A second-chance metric (intersection over the model's area, size-constrained) is also in, accepted only at 80% with a plausible plain fit. Neither rescues the figure's two side photographs: the figure LEANS and the tail is horizontal there, a different pose, not a moved part. The shared-shape metric without the size constraint cheated by shrinking the model into the mask (100% at a looking-down angle), which is why the constraint exists. Pose tolerance beyond a moved part needs articulated alignment, which this is not.

## Phase D: parts, showcase
- [ ] D1 [10] Blocked on a model download: no SAM3 weights on disk (the detection folder is empty) and PartField needs its own environment. The only local part node is Tencent's, which is cloud and paid.
- [x] D2 [16] Turntable GIF (24 frames, 1 MB) and a matched-angle photograph/model pair, both on the showcase page.
- [x] D3 [17] Second subject: a Nike Dunk Low from four store photographs (fits 87 to 98%). The Swoosh, panels and midsole land; the geometry is lumpy and the gallery has no true front.
- [x] (found on the way) Glow is now measured against the whitest thing on the object, not the median: the shoe had "glowed" yellow over 12% of itself.

## Added after the plan: the dorsal plates
- [x] Hybrid engine (`engine=hybrid`, `lib/hybrid.mjs`): Hunyuan3D shape from all views, TripoSplat colour from the front, registered by silhouette and transferred. Plates instead of knobs on the figure's back. Detached-slab removal; ComfyUI liveness probe retries.
- [x] Views carry roles (`views.json`): cardinal sides feed the shape engine; `shapeOnly` views (the same mould in another colourway) never paint; any other key (a close-up) is colour-only. The gold Godzilla Store edition's back three-quarter gave the shape a real back view.
- [x] Every view is matted with the segmentation model before the shape engine sees it. With the flood fill, a caption strip came back as a bar through the figure's head; segmented, registration agreement went 51% to 67%, front fit 66% to 81%, and all four colour photographs painted.
- [x] Showcase republished with the hybrid hero (`53253f90-791`); portfolio pipeline section swapped from the chair to the figure (one photograph vs four, Draco twins, matched-angle card, four-in card, shoe card) and a public copy of the showcase at donflo.vercel.app/notes/photo-to-ar.

## Blocked on the owner
- [ ] [12] AR on a physical handset: open the QR on the AR page with a phone, place the chair, the figure and the shoe, check the relief reads on iOS.
