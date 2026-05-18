"""Box-transport loco-manipulation task — phase 2 of the humanoid-rl project.

The G1 walks to a box sitting on one table, grips it bimanually, walks to
a second table, and places it. Builds on the locomotion velocity-tracking
checkpoints via warm-started weights (see `scripts/train.py`).

Importing this package registers the ``Box-Transport`` gym id so
``gym.make("Box-Transport", cfg=env_cfg)`` works in
``scripts/{train,eval,scene_load}.py``.
"""
import gymnasium as gym

gym.register(
    id="Box-Transport",
    entry_point="tasks.locomotion.box_transport.env:BoxTransportEnv",
    disable_env_checker=True,
    kwargs={},
)
