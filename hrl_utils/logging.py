"""Logging helpers shared across trainers.

Convention: envs may write logging values to either
    (a) flat keys directly on env.extras, e.g. extras["reward/foo"] = tensor
    (b) a nested dict under extras["log"], e.g. extras["log"]["reward/foo"] = float
The trainer's rollout aggregator uses iter_loggable_items() to handle both
transparently.
"""
import torch
from typing import Iterable, Tuple, Any


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Fraction of return variance explained by the value function.

    Returns:
        float in (-inf, 1]. 1 = perfect prediction; 0 = no better than
        predicting the mean; <0 = worse than predicting the mean.
        Returns NaN when var(returns) == 0.
    """
    returns_var = returns.var()
    if returns_var.item() == 0.0:
        return float("nan")
    return float(1.0 - (returns - values).var() / returns_var)


def iter_loggable_items(info: dict) -> Iterable[Tuple[str, Any]]:
    """Yield (key, value) pairs from `info`, transparently flattening
    `info["log"]` if it is a dict.

    Top-level key "log" is consumed entirely when it's a dict; its
    children are yielded at the top level. All other keys pass through
    unchanged.
    """
    for k, v in info.items():
        if k == "log" and isinstance(v, dict):
            yield from v.items()
        else:
            yield k, v
