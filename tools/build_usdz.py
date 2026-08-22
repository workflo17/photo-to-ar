"""The iOS half of the AR handoff: one USDZ, at true scale, with N materials.

Takes the part list the baker wrote and produces a binary-crate USDZ with the
textures packaged inside. Every part becomes its own Mesh prim with its own bound
material, which is simpler and more reliably honoured by AR Quick Look than one mesh
carrying GeomSubsets.

Generalised from ~/product-viz/tools/make_chair_usdz.py, which handled exactly one
material. Run as:  python tools/build_usdz.py <parts.json> <outdir>
"""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdUtils

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = ROOT / "assets" / "materials"

parts_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
doc = json.loads(parts_path.read_text())
name = doc["name"]
parts = doc["parts"]

staging = out_dir / "_usd"
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)


def safe(text):
    """USD prim names are identifiers, not free text."""
    out = "".join(c if c.isalnum() else "_" for c in text)
    return out if out[:1].isalpha() or out[:1] == "_" else f"m_{out}"


PHOTO_DIR = Path(doc["photoDir"]) if doc.get("photoDir") else None


def stage_texture(filename):
    """The photo-colour map lives in the project directory, not the material library."""
    if filename == "__photo__":
        dest = staging / "photo.jpg"
        if not dest.exists():
            shutil.copy(PHOTO_DIR / "photo.jpg", dest)
        return "photo.jpg"
    if filename == "__glow__":
        dest = staging / "glow.jpg"
        if not dest.exists():
            shutil.copy(PHOTO_DIR / "glow.jpg", dest)
        return "glow.jpg"
    if filename == "__detail__":
        # the baked relief map, same atlas as the photo colour; 1024 PNG, left whole
        # because it carries the figure's scales and halving it halves them
        dest = staging / "detail.png"
        if not dest.exists():
            shutil.copy(PHOTO_DIR / "detail.png", dest)
        return "detail.png"
    return _stage_library_texture(filename)


def _stage_library_texture(filename):
    """Copy a map into the staging dir, shrinking normals.

    A 2K normal map is the single biggest thing in the package and AR Quick Look
    downloads the whole file before it shows anything, so the normal is halved. Base
    colour is left alone: that one a shopper actually looks at.
    """
    dest = staging / filename
    if dest.exists():
        return filename
    if filename.endswith("_normal.png"):
        Image.open(MATERIALS / filename).resize((512, 512), Image.LANCZOS).save(dest, optimize=True)
    else:
        shutil.copy(MATERIALS / filename, dest)
    return filename


usdc = staging / f"{name}.usdc"
stage = Usd.Stage.CreateNew(str(usdc))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
# real scale, so an AR placement answers "will it fit" rather than "is it pretty"
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

root = UsdGeom.Xform.Define(stage, f"/{safe(name)}")
stage.SetDefaultPrim(root.GetPrim())
root_path = root.GetPath().pathString


def build_material(index, part):
    mp = f"{root_path}/Materials/{safe(part['material'])}_{index}"
    mat = UsdShade.Material.Define(stage, mp)

    pbr = UsdShade.Shader.Define(stage, f"{mp}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    tint = part["tint"]

    if part["textures"]:
        st_reader = UsdShade.Shader.Define(stage, f"{mp}/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_out = st_reader.ConnectableAPI().GetOutput("result")
        if not st_out:
            st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        def sampler(node, filename, kind, channel=None, scale=None, bias=None):
            sh = UsdShade.Shader.Define(stage, f"{mp}/{node}")
            sh.CreateIdAttr("UsdUVTexture")
            sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{stage_texture(filename)}")
            sh.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            if scale is not None:
                # UsdUVTexture computes value * scale + bias, and this is the ONLY
                # multiply AR Quick Look honours. See the tint note below.
                sh.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            if bias is not None:
                sh.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
            # UVs run past 0..1 by design here: the box projection is measured in
            # metres, so the wrap mode is what makes a weave tile instead of smear
            sh.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            sh.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            if kind == "rgb":
                return sh.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            return sh.CreateOutput(channel, Sdf.ValueTypeNames.Float)

        tex = part["textures"]
        # The tint rides on the sampler's scale, NOT on displayColor.
        #
        # RealityKit and AR Quick Look do not read displayColor or vertex colours at
        # all, so the previous approach silently dropped every colour choice on iPhone:
        # the same chair rendered navy on Android and untinted grey on iOS, with no
        # error anywhere. UsdUVTexture's scale is documented as value * scale + bias and
        # is honoured, so the multiply happens inside the shader instead.
        base = sampler("baseTex", tex["basecolor"], "rgb",
                       scale=(tint[0], tint[1], tint[2], 1.0))
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(base)

        # the photographed glow, as emission; RealityKit honours emissiveColor
        if tex.get("emissive"):
            glow = sampler("glowTex", tex["emissive"], "rgb")
            pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(glow)

        # An 8-bit normal map stores [-1,1] packed into [0,1], and UsdPreviewSurface
        # does NOT unpack it for you. The spec is explicit that scale must be
        # (2,2,2,1) and bias (-1,-1,-1,0) for 8-bit normals. Without them a flat texel
        # (0.5,0.5,1) is read as the normal (0.5,0.5,1) rather than (0,0,1), so every
        # textured surface in the package is systematically tilted toward +X+Y.
        if not tex.get("normal"):
            # the photo colour ships as one baseColor map and nothing else
            pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(part["roughness"])
            mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
            return mat

        nrm = sampler("normTex", tex["normal"], "rgb",
                      scale=(2.0, 2.0, 2.0, 1.0), bias=(-1.0, -1.0, -1.0, 0.0))
        # a normal map is vector data, not colour: without this it is gamma-decoded
        # and the surface looks softly wrong in a way that is hard to place
        stage.GetPrimAtPath(f"{mp}/normTex").GetAttribute("inputs:sourceColorSpace") \
            .Set("raw") if stage.GetPrimAtPath(f"{mp}/normTex").GetAttribute("inputs:sourceColorSpace") \
            else UsdShade.Shader.Get(stage, f"{mp}/normTex") \
            .CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nrm)

        if not tex.get("orm"):
            # photo colour plus its baked relief: no roughness or occlusion maps
            pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(part["roughness"])
            pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(part["metallic"])
            pbr.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
            mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
            return mat

        rough = sampler("roughTex", tex["orm"], "float", "g")
        UsdShade.Shader.Get(stage, f"{mp}/roughTex") \
            .CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(rough)

        occ = sampler("occTex", tex["orm"], "float", "r")
        UsdShade.Shader.Get(stage, f"{mp}/occTex") \
            .CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        pbr.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(occ)

    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))

    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(part["roughness"]) \
        if not part["textures"] else None
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(part["metallic"])
    pbr.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


for i, part in enumerate(parts):
    mesh_path = f"{root_path}/{safe(part['material'])}_{i}"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    pts = part["positions"]
    mesh.CreatePointsAttr([Gf.Vec3f(pts[j], pts[j + 1], pts[j + 2]) for j in range(0, len(pts), 3)])
    nrm = part["normals"]
    mesh.CreateNormalsAttr([Gf.Vec3f(nrm[j], nrm[j + 1], nrm[j + 2]) for j in range(0, len(nrm), 3)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    idx = part["indices"]
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateFaceVertexCountsAttr([3] * (len(idx) // 3))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    if part["textures"] and part["textures"].get("basecolor") == "__photo__":
        mesh.CreateDoubleSidedAttr().Set(True)   # see bake.mjs: splat patches

    uvs = part["uvs"]
    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    # USD's st origin is the bottom-left, glTF's the top-left: flip v on the way in,
    # for every material. The fabric maps tile and never showed it; the photo atlas does.
    st.Set([Gf.Vec2f(uvs[j], 1.0 - uvs[j + 1]) for j in range(0, len(uvs), 2)])

    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh).Bind(build_material(i, part))

stage.GetRootLayer().Save()

usdz_path = out_dir / f"{name}.usdz"
if usdz_path.exists():
    usdz_path.unlink()
ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(usdc)), str(usdz_path))
if not ok or not usdz_path.exists():
    print("failed to package usdz", file=sys.stderr)
    sys.exit(1)

# open it back and confirm it parses, because a corrupt usdz fails silently on a
# phone: the AR button simply does nothing and there is no error anywhere
check = Usd.Stage.Open(str(usdz_path))
if not check or not check.GetDefaultPrim():
    print("packaged usdz has no default prim", file=sys.stderr)
    sys.exit(1)

meshes = sum(1 for p in check.Traverse() if UsdGeom.Mesh(p))
print(f"valid usdz: {usdz_path.name}, {usdz_path.stat().st_size / 1e6:.2f} MB, {meshes} meshes")
shutil.rmtree(staging, ignore_errors=True)
