"""Shared helpers for the evaluator stack.

Anything that's algorithm-aware (PPO vs SAPG/EPO checkpoint handling) or
rendering-aware (frame capture, ray-tracing setup) lives here so the per-task
evaluators in ``tasks/<domain>/<task>/evaluate.py`` stay focused on
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
    and (for SAPG/EPO) appends the leader block's coefficient.

    Three checkpoint formats are auto-detected:
        * rsl_rl PPO:  top-level key ``model_state_dict``
        * our SAPG/EPO: top-level ``model`` containing ``extra_params``
        * our PPO:     top-level ``model`` (no ``extra_params``)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── rsl_rl PPO checkpoint ────────────────────────────────────────────
    # rsl-rl < 4.0  : top-level ``model_state_dict``
    # rsl-rl >= 4.0 : top-level ``actor_state_dict`` (separate actor/critic)
    if "actor_state_dict" in ckpt or "model_state_dict" in ckpt:
        return _load_rsl_rl_policy(ckpt, obs_dim, action_dim, device, deterministic)

    # ── Our PPO / SAPG / EPO checkpoint ──────────────────────────────────
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


# -----------------------------------------------------------------------------
# rsl_rl checkpoint loader
# -----------------------------------------------------------------------------

def _load_rsl_rl_policy(
    ckpt: dict,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    deterministic: bool,
) -> tuple[torch.nn.Module, Callable, bool]:
    """Rebuild a policy from an rsl_rl OnPolicyRunner checkpoint.

    rsl-rl 4.0+ checkpoint structure (what we actually get from the trainer)::

        actor_state_dict:
            obs_normalizer._mean / ._var / ._std / .count  (EmpiricalNormalization)
            distribution.std_param                          (action std, shape (A,))
            mlp.0.weight (H0, obs)  /  mlp.0.bias           (Linear)
            mlp.2.weight (H1, H0)   /  mlp.2.bias           (Linear; idx 1 = activation)
            ...
            mlp.<2N>.weight (A, H_{N-1}) / mlp.<2N>.bias    (output Linear -> action_dim)
        critic_state_dict: same but mlp output dim = 1
        optimizer_state_dict, iter, infos

    We rebuild only the actor (MLP + obs normalization + action std) since eval
    doesn't need the value function. This avoids depending on rsl_rl's internal
    MLPModel/TensorDict construction (which differs between rsl-rl minor versions).
    """
    import torch.nn as nn

    # rsl-rl < 4.0 used a single 'model_state_dict' (legacy ActorCritic). Branch
    # to the legacy loader if we see that instead of the new split layout.
    if "actor_state_dict" not in ckpt and "model_state_dict" in ckpt:
        return _load_rsl_rl_policy_legacy(
            ckpt, obs_dim, action_dim, device, deterministic
        )

    actor_sd = ckpt["actor_state_dict"]

    # Read hidden-layer widths from the mlp.<2k>.weight shapes.
    layer_shapes: list[tuple[int, int]] = []   # (in_dim, out_dim) per Linear
    i = 0
    while f"mlp.{i}.weight" in actor_sd:
        w = actor_sd[f"mlp.{i}.weight"]
        layer_shapes.append((w.shape[1], w.shape[0]))
        i += 2
    if not layer_shapes:
        raise ValueError(
            f"actor_state_dict has no 'mlp.<k>.weight' keys; got {list(actor_sd.keys())[:6]}..."
        )
    hidden_dims = [s[1] for s in layer_shapes[:-1]]

    # Build matching Sequential: Linear, ELU, Linear, ELU, ..., Linear.
    layers: list[nn.Module] = []
    for j, (in_dim, out_dim) in enumerate(layer_shapes):
        layers.append(nn.Linear(in_dim, out_dim))
        if j < len(layer_shapes) - 1:
            layers.append(nn.ELU())   # rsl_rl default activation
    actor_mlp = nn.Sequential(*layers).to(device)

    # Strip 'mlp.' prefix and load just the MLP weights.
    mlp_only = {
        k[len("mlp."):]: v.to(device)
        for k, v in actor_sd.items()
        if k.startswith("mlp.")
    }
    actor_mlp.load_state_dict(mlp_only)
    actor_mlp.eval()

    # Obs normalization: rsl-rl's EmpiricalNormalization stores _mean/_var/_std
    # with shape (1, obs_dim). Apply as (obs - mean) / (std + eps).
    obs_mean = actor_sd["obs_normalizer._mean"].to(device)
    obs_std = actor_sd["obs_normalizer._std"].to(device)

    # Action std (Gaussian distribution param). Stored as direct std when
    # std_type='scalar' (rsl_rl default). Shape (action_dim,).
    action_std = actor_sd["distribution.std_param"].to(device)

    print(
        f"[eval] detected rsl-rl 4.0+ MLPModel actor "
        f"(hidden {hidden_dims}, action_dim {layer_shapes[-1][1]})"
    )

    @torch.no_grad()
    def get_action(obs_raw: torch.Tensor) -> torch.Tensor:
        obs_norm = (obs_raw - obs_mean) / (obs_std + 1e-8)
        mean = actor_mlp(obs_norm)
        if deterministic:
            return mean.clamp(-1.0, 1.0)
        noise = torch.randn_like(mean) * action_std
        return (mean + noise).clamp(-1.0, 1.0)

    return actor_mlp, get_action, False


def _load_rsl_rl_policy_legacy(
    ckpt: dict,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    deterministic: bool,
) -> tuple[torch.nn.Module, Callable, bool]:
    """Loader for rsl-rl < 4.0 checkpoints (single ``model_state_dict`` with
    actor.* / critic.* / std keys).

    Kept as a fallback; current rsl_rl is 4.0+ which uses the split layout
    handled by :func:`_load_rsl_rl_policy`.
    """
    import torch.nn as nn

    sd = ckpt["model_state_dict"]

    def _layer_outs(prefix: str) -> list[int]:
        outs = []
        i = 0
        while f"{prefix}.{i}.weight" in sd:
            outs.append(sd[f"{prefix}.{i}.weight"].shape[0])
            i += 2
        return outs

    actor_outs = _layer_outs("actor")
    if not actor_outs:
        raise ValueError(
            f"legacy rsl_rl ckpt has no 'actor.<k>.weight' keys; got {list(sd.keys())[:6]}..."
        )
    hidden_dims = actor_outs[:-1]

    layers: list[nn.Module] = []
    in_dim = obs_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ELU())
        in_dim = h
    layers.append(nn.Linear(in_dim, action_outs := actor_outs[-1]))
    actor_mlp = nn.Sequential(*layers).to(device)
    actor_mlp.load_state_dict(
        {k[len("actor."):]: v.to(device) for k, v in sd.items() if k.startswith("actor.")}
    )
    actor_mlp.eval()

    action_std = sd.get("std", torch.ones(action_dim)).to(device)

    print(f"[eval] detected legacy rsl-rl < 4.0 ActorCritic (hidden {hidden_dims})")

    @torch.no_grad()
    def get_action(obs_raw: torch.Tensor) -> torch.Tensor:
        mean = actor_mlp(obs_raw)
        if deterministic:
            return mean.clamp(-1.0, 1.0)
        noise = torch.randn_like(mean) * action_std
        return (mean + noise).clamp(-1.0, 1.0)

    return actor_mlp, get_action, False
