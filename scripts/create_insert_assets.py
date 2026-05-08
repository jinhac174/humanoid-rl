"""
Creates USD assets for the Insert task (peg_a, peg_b, block).
Peg geometry faithfully ported from insert_normal.xml compound boxes.

Run once before training:
    ~/IsaacLab/isaaclab.sh -p scripts/create_insert_assets.py
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, Gf

from manipulation.utils.paths import ASSET_ROOT
ASSET_DIR = ASSET_ROOT / "objects" / "insert"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def _add_box(stage, path: str, pos: tuple, half_size: tuple, color: tuple):
    """Add a Cube prim with collision at path, scaled to full_size = 2*half_size."""
    prim = stage.DefinePrim(path, "Cube")
    cube = UsdGeom.Cube(prim)
    cube.CreateSizeAttr(1.0)

    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
    xf.AddScaleOp().Set(Gf.Vec3d(
        half_size[0] * 2.0,
        half_size[1] * 2.0,
        half_size[2] * 2.0,
    ))

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim)


def create_peg_usd(output_path: Path, color: tuple):
    """
    Compound peg — 5 boxes forming an open-top U-cup.
    Geometry from insert_normal.xml (half-extents → full metres).
    Inner opening: 0.06 × 0.06 m. Block prongs: 0.036 × 0.036 m.
    """
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Peg")
    stage.SetDefaultPrim(root.GetPrim())

    # RigidBodyAPI must be on root for IsaacLab RigidObject to find it
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(0.3)

    shapes = [
        ((0.0,   0.0,  0.005), (0.04, 0.04, 0.005)),   # base plate
        ((0.0,  -0.03, 0.04),  (0.04, 0.01, 0.03)),    # -Y wall
        ((0.0,   0.03, 0.04),  (0.04, 0.01, 0.03)),    # +Y wall
        ((-0.03, 0.0,  0.04),  (0.01, 0.04, 0.03)),    # -X wall
        (( 0.03, 0.0,  0.04),  (0.01, 0.04, 0.03)),    # +X wall
    ]

    for i, (pos, half_size) in enumerate(shapes):
        _add_box(stage, f"/Peg/Box_{i}", pos, half_size, color)

    stage.Save()
    print(f"[create_assets] saved: {output_path}")


def create_block_usd(output_path: Path):
    """
    Block — single collision box matching insert_normal.xml:
        collision: half=(0.018, 0.15, 0.018) at local pos (0, 0, 0.018)
    """
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Block")
    stage.SetDefaultPrim(root.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(0.2)

    _add_box(
        stage, "/Block/Collision",
        pos=(0.0, 0.0, 0.018),
        half_size=(0.018, 0.15, 0.018),
        color=(0.65, 0.65, 0.65),
    )

    stage.Save()
    print(f"[create_assets] saved: {output_path}")


if __name__ == "__main__":
    create_peg_usd(ASSET_DIR / "peg_a.usd", color=(0.2, 0.2, 0.2))
    create_peg_usd(ASSET_DIR / "peg_b.usd", color=(0.8, 0.8, 0.8))
    create_block_usd(ASSET_DIR / "block.usd")
    print("[create_assets] all done.")
    simulation_app.close()