"""Partial checkpoint loading for cross-task warm-starting.

Used by ``scripts/train.py`` when ``cfg.checkpoint`` is set: takes a
phase-1 (velocity_tracking) checkpoint and copies as much as possible
into a freshly-built phase-2 (box_transport) trainer. Designed to
work for PPO, SAPG, and EPO without algo-specific branches.

Key trick: the phase-2 observation layout is

    [0:141]    = locomotion proprioception  (identical to phase 1)
    [141:164]  = new manipulation obs (box / target / palm / lifted_flag)

For SAPG/EPO, the network's MLP input has a per-block embedding appended
at the END of the input vector. So phase-1 SAPG input layout is

    [0:141]     loco obs
    [141:173]   embed (32-d)

and phase-2 SAPG input layout is

    [0:141]     loco obs
    [141:164]   new obs (zero-fill at warm-start)
    [164:196]   embed (transferred from phase-1's [141:173])

For plain PPO, embed_dim == 0 and the embed slice is a no-op.

The optimizer state is NOT copied — it has Adam moments shaped for the
old parameter sizes and would be invalid for the new ones. The agent
starts with fresh optimizer state, which is the right thing for a
warm-start anyway (we want the policy to begin adapting to phase-2
gradients without phase-1 momentum dragging it).
"""
from __future__ import annotations

import torch


PHASE1_OBS_DIM = 141   # bytes-identical prefix of the box_transport obs


def warm_start_load(trainer, ckpt_path: str, phase1_obs_dim: int = PHASE1_OBS_DIM) -> int:
    """Partial-load a phase-1 checkpoint into a phase-2 trainer.

    Returns the iteration count from the checkpoint (for logging only;
    the new run still starts iter at 0).
    """
    ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
    old_sd = ckpt["model"]
    new_sd = trainer.agent.network.state_dict()

    copied, partial, skipped = [], [], []

    for key in new_sd:
        if key not in old_sd:
            skipped.append(f"{key} (not in checkpoint)")
            continue
        old, new = old_sd[key], new_sd[key]

        if old.shape == new.shape:
            new_sd[key] = old.clone().to(new.device)
            copied.append(key)
            continue

        # Try the input-layer mapping: 2-D tensor, output dim matches,
        # input dim differs.
        if (
            old.ndim == 2
            and old.shape[0] == new.shape[0]
            and old.shape[1] != new.shape[1]
        ):
            old_in, new_in = old.shape[1], new.shape[1]
            embed_dim = old_in - phase1_obs_dim
            new_obs_added = new_in - old_in
            if embed_dim < 0 or new_obs_added < 0:
                skipped.append(
                    f"{key} (unexpected shape: old={old.shape}, new={new.shape})"
                )
                continue
            new_param = torch.zeros_like(new)
            # Locomotion obs slice [0:phase1_obs_dim] transfers as-is.
            new_param[:, :phase1_obs_dim] = old[:, :phase1_obs_dim]
            # Embed slice (if any) lives at the end of both old and new vectors.
            if embed_dim > 0:
                new_param[:, -embed_dim:] = old[:, -embed_dim:]
            new_sd[key] = new_param
            partial.append(
                f"{key}: {tuple(old.shape)} → {tuple(new.shape)}  "
                f"(loc {phase1_obs_dim} + embed {embed_dim} copied; "
                f"new-obs {new_obs_added} zero-filled)"
            )
            continue

        skipped.append(
            f"{key} (shape mismatch: old={old.shape}, new={new.shape})"
        )

    # Apply the merged state_dict (strict=False is paranoid; new_sd has all keys).
    trainer.agent.network.load_state_dict(new_sd, strict=False)

    # Running obs stats: extend with default values for new dims.
    old_mean = ckpt["obs_mean"]
    old_var = ckpt["obs_var"]
    new_mean = trainer.agent.obs_mean.detach().cpu().clone()
    new_var = trainer.agent.obs_var.detach().cpu().clone()
    if old_mean.shape == new_mean.shape:
        trainer.agent.obs_mean = old_mean.to(trainer.device)
        trainer.agent.obs_var = old_var.to(trainer.device)
        trainer.agent.obs_count = ckpt["obs_count"].to(trainer.device)
        stat_note = "obs_mean/var/count: full copy"
    else:
        d = old_mean.shape[0]
        new_mean[:d] = old_mean
        new_var[:d] = old_var
        trainer.agent.obs_mean = new_mean.to(trainer.device)
        trainer.agent.obs_var = new_var.to(trainer.device)
        trainer.agent.obs_count = ckpt["obs_count"].to(trainer.device)
        stat_note = (
            f"obs_mean/var: first {d} dims copied, "
            f"new dims at default (0/1); count copied"
        )

    val_note = "value_mean_std: not in ckpt"
    if trainer.agent.normalize_value and "value_mean_std" in ckpt:
        trainer.agent.value_mean_std.load_state_dict(ckpt["value_mean_std"])
        val_note = "value_mean_std: full copy"

    print(f"[warm_start] source: {ckpt_path}  (was iter {ckpt.get('iteration', '?')})")
    print(f"[warm_start] {len(copied)} tensors copied verbatim:")
    for k in copied:
        print(f"    ✓ {k}")
    if partial:
        print(f"[warm_start] {len(partial)} tensors partially copied:")
        for p in partial:
            print(f"    ⊕ {p}")
    if skipped:
        print(f"[warm_start] {len(skipped)} tensors skipped:")
        for s in skipped:
            print(f"    ✗ {s}")
    print(f"[warm_start] {stat_note}")
    print(f"[warm_start] {val_note}")
    print(f"[warm_start] optimizer: NOT copied (fresh state)")
    return int(ckpt.get("iteration", 0))
