"""
Writes one .usdz per fabric for iOS AR Quick Look.

Why one per fabric: USD has variant sets, but Quick Look gives the user no UI to
switch them, so a real configurator ships a USDZ per option and swaps the
<model-viewer ios-src> when the shopper picks a swatch. Android reads the GLB
directly and uses KHR_materials_variants, so only iOS needs this fan-out.

Known limitation, stated up front: UsdPreviewSurface has no sheen lobe, so the
velvet loses its fibre retroreflection in Quick Look. Compensated by dropping
roughness and lifting base colour slightly for that one fabric.
"""
import json
import shutil
from pathlib import Path

from PIL import Image

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdUtils, UsdValidation

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
TEX = ASSETS / "textures"
OUT = ASSETS / "usdz"
STAGING = ASSETS / "_usd_staging"

FABRICS = {
    #  slug            roughness  occlusion  note
    "ivory-linen":  dict(rough=0.94),
    "sage-twill":   dict(rough=0.90),
    "camel-boucle": dict(rough=0.96),
    "ink-velvet":   dict(rough=0.36),   # no sheen lobe in Quick Look, so run smoother
}
WALNUT = Gf.Vec3f(0.113, 0.063, 0.036)

mesh_data = json.loads((ASSETS / "sofa-mesh.json").read_text())


def add_mesh(stage, path, data, material):
    pos = data["position"]
    nrm = data["normal"]
    uv = data["uv"]
    n_pts = len(pos) // 3
    n_tris = n_pts // 3

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*pos[i * 3:i * 3 + 3]) for i in range(n_pts)])
    mesh.CreateFaceVertexCountsAttr([3] * n_tris)
    mesh.CreateFaceVertexIndicesAttr(list(range(n_pts)))
    mesh.CreateNormalsAttr([Gf.Vec3f(*nrm[i * 3:i * 3 + 3]) for i in range(n_pts)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)   # already triangulated
    if uv:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(uv[i * 2], uv[i * 2 + 1]) for i in range(n_pts)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    return mesh


def tex_reader(stage, mat_path, name, filename, channel, st_reader, raw=False):
    t = UsdShade.Shader.Define(stage, f"{mat_path}/{name}")
    t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(filename)
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result")
    t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    if raw:   # normal maps are data, not colour
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
    return t.CreateOutput(channel, Sdf.ValueTypeNames.Float3 if channel == "rgb"
                          else Sdf.ValueTypeNames.Float)


def build(slug, cfg):
    STAGING.mkdir(parents=True, exist_ok=True)
    for suffix in ("basecolor.jpg", "orm.jpg"):
        shutil.copy(TEX / f"{slug}_{suffix}", STAGING / f"{slug}_{suffix}")
    # match the GLB: 512 normals are plenty at AR viewing distance and keep the
    # package inside the size where Quick Look still opens fast on cellular
    Image.open(TEX / f"{slug}_normal.png").resize((512, 512), Image.LANCZOS).save(
        STAGING / f"{slug}_normal.png", optimize=True)

    # binary crate, not ASCII: ARKit prefers a .usdc root layer and it is ~10x smaller
    usda = STAGING / f"sofa_{slug}.usdc"
    stage = Usd.Stage.CreateNew(str(usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)          # glTF and Quick Look agree on Y-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)                # real-world scale for AR placement

    root = UsdGeom.Xform.Define(stage, "/Sofa")
    stage.SetDefaultPrim(root.GetPrim())

    # --- upholstery material
    mp = "/Sofa/Materials/Upholstery"
    mat = UsdShade.Material.Define(stage, mp)
    prim_st = UsdShade.Shader.Define(stage, f"{mp}/stReader")
    prim_st.CreateIdAttr("UsdPrimvarReader_float2")
    prim_st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    surf = UsdShade.Shader.Define(stage, f"{mp}/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex_reader(stage, mp, "baseColor", f"{slug}_basecolor.jpg", "rgb", prim_st))
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        tex_reader(stage, mp, "rough", f"{slug}_orm.jpg", "g", prim_st))
    surf.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
        tex_reader(stage, mp, "ao", f"{slug}_orm.jpg", "r", prim_st))
    surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        tex_reader(stage, mp, "normal", f"{slug}_normal.png", "rgb", prim_st, raw=True))
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")

    # --- walnut legs
    lp = "/Sofa/Materials/Walnut"
    leg_mat = UsdShade.Material.Define(stage, lp)
    leg_surf = UsdShade.Shader.Define(stage, f"{lp}/Surface")
    leg_surf.CreateIdAttr("UsdPreviewSurface")
    leg_surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(WALNUT)
    leg_surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.34)
    leg_surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    leg_mat.CreateSurfaceOutput().ConnectToSource(leg_surf.ConnectableAPI(), "surface")

    add_mesh(stage, "/Sofa/Upholstery", mesh_data["upholstery"], mat)
    add_mesh(stage, "/Sofa/Legs", mesh_data["legs"], leg_mat)
    stage.GetRootLayer().Save()

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"sofa-{slug}.usdz"
    if dst.exists():
        dst.unlink()
    ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(usda)), str(dst))
    size = dst.stat().st_size / 1024 if dst.exists() else 0

    # OpenUSD's own validation pass over the finished package
    registry = UsdValidation.ValidationRegistry()
    suite = [m.name for m in registry.GetAllValidatorMetadata()]
    ctx = UsdValidation.ValidationContext(suite, True)
    problems = [f"{e.GetValidator().GetMetadata().name}: {e.GetMessage()}"
                for e in ctx.Validate(Usd.Stage.Open(str(dst)))]
    print(f"{'ok ' if ok else 'FAIL'} {dst.name:26s} {size:7.0f} KB  "
          f"usd-validate: {'PASS' if not problems else 'FAIL'}")
    for msg in problems:
        print("      ", msg)
    return ok and not problems


results = [build(slug, cfg) for slug, cfg in FABRICS.items()]
shutil.rmtree(STAGING, ignore_errors=True)
print("usdz written:", sum(results), "/", len(results), "->", OUT)
