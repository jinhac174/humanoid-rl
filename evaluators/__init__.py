"""Per-task evaluators.

Mirror of ``algos/<algo>/trainer.py``: each task in
``tasks/<domain>/<task>/evaluate.py`` defines an evaluator class that
subclasses :class:`evaluators.base.BaseEvaluator`. ``scripts/eval.py``
loads the right class via two fields in the task yaml::

    evaluator_module: tasks.manipulation.reorient.evaluate
    evaluator_class:  ReorientEvaluator

Tasks that don't need any task-specific eval logic can omit those fields
and fall back to the generic :class:`BaseEvaluator`.
"""

from evaluators.base import BaseEvaluator

__all__ = ["BaseEvaluator"]
