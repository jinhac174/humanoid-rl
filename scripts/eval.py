"""Generic evaluation entry point.

Loads any checkpoint (PPO, SAPG, EPO), instantiates the task's evaluator
class, and runs the eval loop. All task-specific logic lives in the
evaluator class declared by the task yaml::

    # configs/task/reorient.yaml
    evaluator_module: tasks.manipulation.reorient.evaluate
    evaluator_class:  ReorientEvaluator

Tasks that need no special eval behaviour can omit those fields and the
script falls back to :class:`evaluators.base.BaseEvaluator`.

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/eval.py task=reorient \\
        checkpoint=outputs/reorient/sapg/run_00/checkpoints/model_21000.pt

    # Publication quality (path-traced, 1080p)
    ~/IsaacLab/isaaclab.sh -p scripts/eval.py task=reorient \\
        checkpoint=... raytraced=true video_width=1920 video_height=1080
"""
import importlib

import hydra
from omegaconf import DictConfig

from isaaclab.app import AppLauncher


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    # AppLauncher must be the first IsaacLab/Isaac Sim import. Cameras are
    # required for video rendering.
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    # Resolve the task's evaluator class. Falling back to BaseEvaluator means
    # tasks that need no overrides never have to ship an evaluate.py file.
    evaluator_module = getattr(cfg.task, "evaluator_module", None)
    evaluator_class = getattr(cfg.task, "evaluator_class", None)
    if evaluator_module and evaluator_class:
        module = importlib.import_module(evaluator_module)
        EvaluatorClass = getattr(module, evaluator_class)
    else:
        from evaluators.base import BaseEvaluator
        EvaluatorClass = BaseEvaluator

    evaluator = EvaluatorClass(cfg)
    try:
        # Renderer setup must happen BEFORE the scene is built so carb settings
        # are picked up by Omniverse on stage creation.
        evaluator.setup_renderer()
        evaluator.build_env()
        evaluator.load_policy()
        evaluator.run()
    finally:
        evaluator.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
