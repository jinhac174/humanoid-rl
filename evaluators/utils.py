"""Shared helpers for the evaluator stack.

Anything that's algorithm-aware (PPO vs SAPG/EPO checkpoint handling) or
rendering-aware (frame capture, ray-tracing setup) lives here so the per-task
evaluators in ``manipulation/tasks/.../evaluate.py`` stay focused on
task-specific eval behaviour.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------

def grab_frame(env) -> np.ndarray:
    """Return the current rendered frame as ``(H, W, 3) uint8``.

    Handles the variations IsaacLab's render() can return: torch tensor,
    numpy array, batched (1, H, W, 3) shape, and float pixels.
    """
    frame = env.render()
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def setup_raytracing(spp: int) -> bool:
    """Switch the renderer to path tracing. Returns True on success.

    Must be called BEFORE the scene is built (i.e. before ``gym.make``).
    Falls back silently to rasterization if the carb settings are missing.
    """
    try:
        import carb
        settings = carb.settings.get_settings()
        settings.set_string("/rtx/rendermode", "PathTracing")
        settings.set_int("/rtx/pathtracing/spp", spp)
        settings.set_int("/rtx/pathtracing/totalSpp", spp)
        settings.set_int("/rtx/pathtracing/maxBounces", 4)
        settings.set_bool("/rtx/pathtracing/enabled", True)
        settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)
        return True
    except Exception as e:
        print(f"[eval] ray-tracing setup failed: {e}, falling back to rasterization")
        return False


# -----------------------------------------------------------------------------
# Checkpoint -> policy reconstruction
# -----------------------------------------------------------------------------

def _infer_hidden_dims(sd: dict, prefix: str = "trunk") -> list[int]:
    """Read a state-dict and return the trunk's hidden layer widths.

    Both the PPO and SAPG networks build the trunk as
    ``[Linear, Act, Linear, Act, ...]``, so trunk weights live at indices
    0, 2, 4, ... — we step by 2 and stop when the weight key disappears.
    """
    hidden_dims = []
    i = 0
    while f"{prefix}.{i}.weight" in sd:
        hidden_dims.append(sd[f"{prefix}.{i}.weight"].shape[0])
        i += 2
    return hidden_dims


def load_policy_from_checkpoint(
    ckpt_path: Path,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    deterministic: bool,
) -> tuple[torch.nn.Module, Callable, bool]:
    """Auto-detect the algo from a checkpoint and rebuild the corresponding
    network plus an action-selection closure.

    Returns ``(network, get_action, is_sapg)`` where ``get_action(obs_raw)``
    takes a raw observation tensor of shape ``(N, obs_dim)`` and returns an
    action tensor clamped to [-1, 1]. The closure handles obs normalization
    using the obs_mean / obs_var from the checkpoint and (for SAPG/EPO)
    appends the leader block's coefficient.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"]

    obs_mean = ckpt.get("obs_mean", torch.zeros(obs_dim)).to(device)
    obs_var = ckpt.get("obs_var", torch.ones(obs_dim)).to(device)

    is_sapg = "extra_params" in sd

    if is_sapg:
        from algos.sapg.network import SAPGActorCritic

        num_blocks = sd["extra_params"].shape[0]
        extra_param_size = sd["extra_params"].shape[1]
        # Match SAPG's training-time block IDs: linspace(50, 0, num_blocks).
        block_ids = torch.linspace(50.0, 0.0, num_blocks, device=device)
        hidden_dims = _infer_hidden_dims(sd, "trunk")

        network = SAPGActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="elu",
            block_ids=block_ids,
            extra_param_size=extra_param_size,
        ).to(device)
        network.load_state_dict(sd)
        network.eval()

        # Eval always uses the leader (block 0) — same convention as training.
        leader_coef = block_ids[0:1].reshape(1, 1)
        print(f"[eval] detected SAPG/EPO network ({num_blocks} blocks)")

        def normalize(obs_raw: torch.Tensor) -> torch.Tensor:
            obs = obs_raw.clamp(-100.0, 100.0)
            return ((obs - obs_mean) / (obs_var.sqrt() + 1e-8)).clamp(-10.0, 10.0)

        def get_action(obs_raw: torch.Tensor) -> torch.Tensor:
            obs_norm = normalize(obs_raw)
            net_input = torch.cat([obs_norm, leader_coef], dim=1)
            if deterministic:
                mu, _, _ = network.forward(net_input)
                return mu.clamp(-1.0, 1.0)
            action, _, _, _, _, _ = network.get_action_and_value(net_input)
            return action.clamp(-1.0, 1.0)

    else:
        from algos.ppo.network import ActorCritic

        hidden_dims = _infer_hidden_dims(sd, "trunk")
        network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            shared=True,
            hidden_dims=hidden_dims,
            activation="elu",
            init_noise_std=1.0,
            use_tanh=False,
        ).to(device)
        network.load_state_dict(sd)
        network.eval()
        print("[eval] detected PPO network")

        def normalize(obs_raw: torch.Tensor) -> torch.Tensor:
            obs = obs_raw.clamp(-100.0, 100.0)
            return ((obs - obs_mean) / (obs_var.sqrt() + 1e-8)).clamp(-10.0, 10.0)

        def get_action(obs_raw: torch.Tensor) -> torch.Tensor:
            obs_norm = normalize(obs_raw)
            if deterministic:
                features = network.trunk(obs_norm)
                action = network.actor_head(features)
                return action.clamp(-1.0, 1.0)
            action, _, _, _, _ = network.get_action_and_value(obs_norm)
            return action.clamp(-1.0, 1.0)

    return network, get_action, is_sapg
