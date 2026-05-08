"""
Reorient task -- G1 bimanual cuboid reorientation.

Port of SAPG's Two-Arms Reorientation (isaacgymenvs AllegroKukaTwoArmsReorientation)
adapted for the Unitree G1 + Dex3 humanoid in IsaacLab 2.3.

Importing this package registers the `Reorient` gym id so that
`gym.make("Reorient", cfg=env_cfg)` works in scripts/train.py and
scripts/scene_load.py. The entry point is a string so that isaaclab
and torch are not imported until gym.make is actually called.
"""
import gymnasium as gym

gym.register(
    id="Reorient",
    entry_point="manipulation.tasks.reorient.env:ReorientEnv",
    disable_env_checker=True,
    kwargs={},
)