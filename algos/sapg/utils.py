"""
SAPG utility functions.

Ported from rl_games/common/custom_utils.py in the SAPG repo.
"""
import torch
import numpy as np


def filter_leader(
    val: torch.Tensor,
    orig_len: int,
    repeat_idxs: list[int],
    num_blocks: int,
) -> torch.Tensor:
    """
    Leader-follower filtering for SAPG batch augmentation.

    For the leader (repeat_idx=0): keep the FULL original batch.
    For follower block k (repeat_idx=k): keep only block k-1's slice.

    This ensures the leader trains on all its own data plus a filtered
    subset from each sampled follower, preventing the followers from
    overwhelming the leader's gradient signal.

    Ported verbatim from SAPG repo custom_utils.py::filter_leader.

    Args:
        val: concatenated tensor of shape (num_repeats * orig_len, ...) or
             (dim0, num_repeats * orig_len, ...) if dim0 > 1
        orig_len: length of one copy of the original batch
        repeat_idxs: list of block indices [0, k1, k2, ...] where 0 = leader
        num_blocks: total number of SAPG blocks
    """
    bsize = orig_len // num_blocks
    if val.dim() >= 2 and val.shape[0] > 1 and val.shape[0] != len(repeat_idxs) * orig_len:
        # Axis-1 case (e.g. RNN states): filter along dim=1
        filtered = []
        for i, idx in enumerate(repeat_idxs):
            if idx == 0:
                filtered.append(val[:, i * orig_len : (i + 1) * orig_len])
            else:
                start = i * orig_len + (idx - 1) * bsize
                end = i * orig_len + idx * bsize
                filtered.append(val[:, start:end])
        return torch.cat(filtered, dim=1)
    else:
        # Axis-0 case (most tensors)
        filtered = []
        for i, idx in enumerate(repeat_idxs):
            if idx == 0:
                filtered.append(val[i * orig_len : (i + 1) * orig_len])
            else:
                start = i * orig_len + (idx - 1) * bsize
                end = i * orig_len + idx * bsize
                filtered.append(val[start:end])
        return torch.cat(filtered, dim=0)


def swap_and_flatten01(arr: torch.Tensor) -> torch.Tensor:
    """Swap axes 0 and 1, then flatten them into a single axis.

    (T, N, ...) → (N, T, ...) → (N*T, ...)

    Ported from SAPG repo custom_utils.py::swap_and_flatten01.
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])