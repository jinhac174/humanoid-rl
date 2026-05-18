"""Dump structure of a torch checkpoint. Usage:
    ~/IsaacLab/isaaclab.sh -p tools/inspect_ckpt.py <path-to-ckpt.pt>
"""
import sys
import torch


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "outputs/velocity_tracking/rsl_rl_ppo/rsl_rl_ppo_03/checkpoints/model_1499.pt"
    )
    c = torch.load(path, map_location="cpu", weights_only=False)
    print(f"=== {path} ===")
    print(f"TYPE: {type(c).__name__}")
    print(f"TOP KEYS ({len(c)}): {list(c.keys())}")
    print()
    for k, v in c.items():
        if hasattr(v, "keys"):
            ks = list(v.keys())
            print(f"  {k!r}  dict with {len(ks)} keys:")
            for sk in ks[:20]:
                sv = v[sk]
                shape = tuple(sv.shape) if hasattr(sv, "shape") else type(sv).__name__
                print(f"      {sk}  -> {shape}")
            if len(ks) > 20:
                print(f"      ... ({len(ks)-20} more)")
        elif hasattr(v, "shape"):
            print(f"  {k!r}  tensor {tuple(v.shape)}")
        else:
            print(f"  {k!r}  {type(v).__name__} = {v!r}")


if __name__ == "__main__":
    main()
