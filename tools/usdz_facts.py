"""Report what a built USDZ actually says, as JSON, so tests can assert on it.

Every fact here corresponds to a bug that shipped silently. A USDZ that is wrong in
these ways still opens, still validates, and still renders: it just renders the wrong
colour or the wrong lighting, on iOS only, with no error anywhere. The only way to
catch that from a build machine with no iPhone attached is to read the package back and
check what it claims.

Usage: python tools/usdz_facts.py <file.usdz>
"""
import json
import sys

from pxr import Usd, UsdGeom, UsdShade

stage = Usd.Stage.Open(sys.argv[1])
facts = {
    "opens": bool(stage),
    "defaultPrim": bool(stage and stage.GetDefaultPrim()),
    "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
    "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
    "meshes": 0,
    # AR Quick Look reads none of these, so an authored one means a lost colour
    "meshesWithAuthoredDisplayColor": 0,
    "baseTexScales": [],
    "normTexScales": [],
    "normTexBiases": [],
    "normTexColorSpaces": [],
    "primNames": [],
}

for prim in stage.Traverse():
    if UsdGeom.Mesh(prim):
        facts["meshes"] += 1
        facts["primNames"].append(prim.GetName())
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
        if pv and pv.HasAuthoredValue():
            facts["meshesWithAuthoredDisplayColor"] += 1

    shader = UsdShade.Shader(prim)
    if not shader or shader.GetIdAttr().Get() != "UsdUVTexture":
        continue
    name = prim.GetName()
    get = lambda k: (lambda i: list(i.Get()) if i and i.Get() is not None else None)(shader.GetInput(k))
    if name == "baseTex":
        facts["baseTexScales"].append(get("scale"))
    if name == "normTex":
        f = shader.GetInput("file")
        facts.setdefault("normTexFiles", []).append(str(f.Get()) if f and f.Get() else None)
        facts["normTexScales"].append(get("scale"))
        facts["normTexBiases"].append(get("bias"))
        cs = shader.GetInput("sourceColorSpace")
        facts["normTexColorSpaces"].append(str(cs.Get()) if cs and cs.Get() else None)

print(json.dumps(facts))
