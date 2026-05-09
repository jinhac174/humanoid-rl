"""Locomotion tasks (free-base, full-body action space).

Placeholder — no tasks yet. When the first locomotion task lands, mirror
the manipulation layout::

    tasks/locomotion/
    ├── __init__.py            # from . import <task_name>
    └── <task_name>/
        ├── __init__.py        # gym.register(id="...", entry_point=...)
        ├── env_cfg.py
        ├── env.py
        ├── observations.py
        ├── rewards.py
        ├── terminations.py
        ├── events.py
        └── evaluate.py        # optional task-specific evaluator
"""
