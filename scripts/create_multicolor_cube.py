"""
Create a multi-color cube USD with 6 distinct face colors.
Run once:
    ~/IsaacLab/isaaclab.sh -p scripts/create_multicolor_cube.py
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from pathlib import Path

HALF = 0.025  # 0.05m cube, half-extent

# Face definitions: (name, 4 corner points, color RGB)
# Winding order: outward-facing normal via right-hand rule
FACES = {
    "face_xp": {
        "pts": [
            Gf.Vec3f(HALF, -HALF, -HALF),
            Gf.Vec3f(HALF, HALF, -HALF),
            Gf.Vec3f(HALF, HALF, HALF),
            Gf.Vec3f(HALF, -HALF, HALF),
        ],
        "color": (1.0, 0.0, 0.0),  # red
    },
    "face_xn": {
        "pts": [
            Gf.Vec3f(-HALF, -HALF, HALF),
            Gf.Vec3f(-HALF, HALF, HALF),
            Gf.Vec3f(-HALF, HALF, -HALF),
            Gf.Vec3f(-HALF, -HALF, -HALF),
        ],
        "color": (0.0, 1.0, 1.0),  # cyan
    },
    "face_yp": {
        "pts": [
            Gf.Vec3f(-HALF, HALF, -HALF),
            Gf.Vec3f(-HALF, HALF, HALF),
            Gf.Vec3f(HALF, HALF, HALF),
            Gf.Vec3f(HALF, HALF, -HALF),
        ],
        "color": (0.0, 1.0, 0.0),  # green
    },
    "face_yn": {
        "pts": [
            Gf.Vec3f(HALF, -HALF, -HALF),
            Gf.Vec3f(HALF, -HALF, HALF),
            Gf.Vec3f(-HALF, -HALF, HALF),
            Gf.Vec3f(-HALF, -HALF, -HALF),
        ],
        "color": (1.0, 0.0, 1.0),  # magenta
    },
    "face_zp": {
        "pts": [
            Gf.Vec3f(-HALF, -HALF, HALF),
            Gf.Vec3f(HALF, -HALF, HALF),
            Gf.Vec3f(HALF, HALF, HALF),
            Gf.Vec3f(-HALF, HALF, HALF),
        ],
        "color": (0.0, 0.0, 1.0),  # blue
    },
    "face_zn": {
        "pts": [
            Gf.Vec3f(-HALF, HALF, -HALF),
            Gf.Vec3f(HALF, HALF, -HALF),
            Gf.Vec3f(HALF, -HALF, -HALF),
            Gf.Vec3f(-HALF, -HALF, -HALF),
        ],
        "color": (1.0, 1.0, 0.0),  # yellow
    },
}


def make_material(stage, path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def make_face_mesh(stage, path, pts):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def create_cube(output_path, with_physics):
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Cube")
    stage.SetDefaultPrim(root.GetPrim())

    # RigidBodyAPI is always needed (IsaacLab requires it for RigidObjectCfg)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

    if with_physics:
        mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass_api.CreateMassAttr().Set(0.1)

        # Invisible collision box (physics only)
        col = UsdGeom.Cube.Define(stage, "/Cube/Collision")
        col.CreateSizeAttr(2.0 * HALF)
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        col.CreatePurposeAttr(UsdGeom.Tokens.guide)

    # Materials
    UsdGeom.Scope.Define(stage, "/Cube/Materials")
    UsdGeom.Xform.Define(stage, "/Cube/Visuals")

    for name, face in FACES.items():
        mat = make_material(stage, f"/Cube/Materials/{name}", face["color"])
        mesh = make_face_mesh(stage, f"/Cube/Visuals/{name}", face["pts"])
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

    stage.Save()
    print(f"[create_multicolor_cube] saved: {output_path}")


if __name__ == "__main__":
    out_dir = Path("assets/objects")
    out_dir.mkdir(parents=True, exist_ok=True)

    create_cube(out_dir / "cube_multicolor.usd", with_physics=True)
    create_cube(out_dir / "cube_multicolor_goal.usd", with_physics=False)

    print("[create_multicolor_cube] done. Files:")
    print(f"  assets/objects/cube_multicolor.usd      (physics cube)")
    print(f"  assets/objects/cube_multicolor_goal.usd  (visual-only goal)")

    simulation_app.close()