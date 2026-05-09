"""Task tree.

Two domains live here: ``tasks.manipulation`` (fixed-base) and
``tasks.locomotion`` (free-base, not yet populated). Importing this package
auto-imports both sub-packages so every task's ``gym.register(...)`` runs
and ``scripts/{train,eval,scene_load}.py`` can ``gym.make`` any task by id.
"""
from . import manipulation  # noqa: F401
from . import locomotion  # noqa: F401
