"""Velocity-tracking locomotion task — direct-style port of Isaac-Velocity-Flat-G1-v0.

Importing this package registers the ``Velocity-Tracking`` gym id so that
``gym.make("Velocity-Tracking", cfg=env_cfg)`` works in
``scripts/{train,eval,scene_load}.py``.
"""
import gymnasium as gym

gym.register(
    id="Velocity-Tracking",
    entry_point="tasks.locomotion.velocity_tracking.env:VelocityTrackingEnv",
    disable_env_checker=True,
    kwargs={},
)
