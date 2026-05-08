"""
Eval script. Loads any checkpoint (PPO, SAPG, EPO), runs the policy, records video.

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/eval.py task=reorient \
        checkpoint=outputs/reorient/sapg/run_00/checkpoints/model_21000.pt

    # Publication quality (ray-traced, 1080p):
    ~/IsaacLab/isaaclab.sh -p scripts/eval.py task=reorient \
        checkpoint=... raytraced=true video_width=1920 video_height=1080

    # 4K ray-traced (slow but beautiful):
    ~/IsaacLab/isaaclab.sh -p scripts/eval.py task=reorient \
        checkpoint=... raytraced=true video_width=3840 video_height=2160 spp=128
"""
import importlib
import torch
import hydra
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import manipulation.tasks

    # -- Ray-tracing setup (must be before scene creation) --
    use_raytracing = getattr(cfg, "raytraced", False)
    spp = getattr(cfg, "spp", 64)
    if use_raytracing:
        try:
            import carb
            settings = carb.settings.get_settings()
            # Enable path tracing
            settings.set_string("/rtx/rendermode", "PathTracing")
            settings.set_int("/rtx/pathtracing/spp", spp)
            settings.set_int("/rtx/pathtracing/totalSpp", spp)
            settings.set_int("/rtx/pathtracing/maxBounces", 4)
            settings.set_bool("/rtx/pathtracing/enabled", True)
            # Denoiser for cleaner output at lower spp
            settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)
            print(f"[eval] ray-tracing ENABLED (spp={spp})")
        except Exception as e:
            print(f"[eval] ray-tracing setup failed: {e}, falling back to rasterization")
            use_raytracing = False
    else:
        print("[eval] rasterization mode (use raytraced=true for publication quality)")

    checkpoint_path = Path(cfg.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    eval_dir = checkpoint_path.parent.parent / "eval" / checkpoint_path.stem
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_tol = cfg.eval_success_tolerance
    eval_steps = cfg.eval_success_steps
    hold_frames = cfg.hold_frames

    print(f"[eval] checkpoint:         {checkpoint_path}")
    print(f"[eval] output:             {eval_dir}")
    print(f"[eval] resolution:         {cfg.video_width}x{cfg.video_height}")
    print(f"[eval] success_tolerance:  {eval_tol} (training uses 0.075)")
    print(f"[eval] success_steps:      {eval_steps} (must hold pose for {eval_steps} steps)")
    print(f"[eval] hold_frames:        {hold_frames} (video pause after success)")

    SCRIPT_KEYS = {"gym_id", "log_name", "env_cfg_module", "env_cfg_class",
                   "cameras", "viewer"}

    cameras = cfg.task.cameras
    first_cam = next(iter(cameras.values()))

    # -- Build env (1 env for eval) --
    module = importlib.import_module(cfg.task.env_cfg_module)
    EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1

    task_dict = OmegaConf.to_container(cfg.task, resolve=True)
    for key, val in task_dict.items():
        if key in SCRIPT_KEYS:
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, val)

    # Override with eval-specific tighter criteria
    env_cfg.success_tolerance = eval_tol
    env_cfg.success_steps = eval_steps

    # Eval-only overrides:
    # - episode_length_s: long videos so we can actually watch the policy
    # - max_consecutive_successes: training uses 50 so episodes terminate
    #   after the policy has demonstrated enough success. For eval at loose
    #   tolerance the policy can hit 50 successes in ~1 second, ending the
    #   video. Effectively disable this termination during eval.
    env_cfg.episode_length_s = 20.0
    env_cfg.max_consecutive_successes = 10**9

    env_cfg.viewer.resolution = (cfg.video_width, cfg.video_height)
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = tuple(first_cam.eye)
    env_cfg.viewer.lookat = tuple(first_cam.lookat)

    env = gym.make(cfg.task.gym_id, cfg=env_cfg, render_mode="rgb_array")

    obs_dim = env.unwrapped.single_observation_space["policy"].shape[0]
    action_dim = env.unwrapped.single_action_space.shape[0]
    device = env.unwrapped.device

    # -- Load checkpoint and build network --
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    obs_mean = ckpt.get("obs_mean", torch.zeros(obs_dim)).to(device)
    obs_var = ckpt.get("obs_var", torch.ones(obs_dim)).to(device)

    is_sapg = "extra_params" in sd

    if is_sapg:
        from manipulation.algos.sapg.network import SAPGActorCritic

        num_blocks = sd["extra_params"].shape[0]
        extra_param_size = sd["extra_params"].shape[1]
        block_ids = torch.linspace(50.0, 0.0, num_blocks, device=device)

        hidden_dims = []
        i = 0
        while f"trunk.{i}.weight" in sd:
            hidden_dims.append(sd[f"trunk.{i}.weight"].shape[0])
            i += 2

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

        leader_coef = block_ids[0:1].reshape(1, 1)
        print(f"[eval] detected SAPG/EPO network ({num_blocks} blocks)")
    else:
        from manipulation.algos.ppo.network import ActorCritic

        hidden_dims = []
        i = 0
        while f"trunk.{i}.weight" in sd:
            hidden_dims.append(sd[f"trunk.{i}.weight"].shape[0])
            i += 2

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

        leader_coef = None
        print(f"[eval] detected PPO network")

    def normalize_obs(obs_raw):
        obs = obs_raw.clamp(-100.0, 100.0)
        return ((obs - obs_mean) / (obs_var.sqrt() + 1e-8)).clamp(-10.0, 10.0)

    def get_action(obs_raw):
        obs_norm = normalize_obs(obs_raw)
        if is_sapg:
            net_input = torch.cat([obs_norm, leader_coef], dim=1)
            if cfg.deterministic:
                mu, _, _ = network.forward(net_input)
                return mu.clamp(-1.0, 1.0)
            else:
                action, _, _, _, _, _ = network.get_action_and_value(net_input)
                return action.clamp(-1.0, 1.0)
        else:
            if cfg.deterministic:
                features = network.trunk(obs_norm)
                action = network.actor_head(features)
                return action.clamp(-1.0, 1.0)
            else:
                action, _, _, _, _ = network.get_action_and_value(obs_norm)
                return action.clamp(-1.0, 1.0)

    def record_frame(writers_dict):
        if use_raytracing:
            # Extra render steps for path tracing convergence
            for _ in range(3):
                env.render()
        for cam_name, cam_cfg_i in cameras.items():
            sim.set_camera_view(eye=list(cam_cfg_i.eye), target=list(cam_cfg_i.lookat))
            frame = _get_frame(env)
            writers_dict[cam_name].append_data(frame)

    sim = env.unwrapped.sim
    unwrapped = env.unwrapped

    # -- Prime renderer --
    obs_dict, _ = env.reset(seed=0)
    prime_steps = 100 if use_raytracing else 50
    for _ in range(prime_steps):
        sim.step()
    env.render()
    for _ in range(10):
        sim.step()

    # -- Episodes --
    for ep in range(cfg.num_episodes):
        obs_dict, _ = env.reset(seed=ep)
        obs_raw = obs_dict["policy"]

        max_steps = int(unwrapped.max_episode_length)
        total_reward = 0.0
        successes = 0
        step = 0

        writers = {}
        for cam_name in cameras:
            path = eval_dir / f"{cam_name}_ep{ep:03d}.mp4"
            writers[cam_name] = imageio.get_writer(str(path), fps=cfg.video_fps)

        record_frame(writers)

        # Cooldown rate-limits how often goals can respawn after a success.
        # On a near-goal step, env queues a respawn (consumed by next step's
        # _pre_physics_step) -- we let that happen naturally. We just suppress
        # any FURTHER respawn triggers for `hold_frames` steps, so the goal
        # cube doesn't strobe when the imprecise policy hovers near it.
        # No hold, no static moment, just continuous motion with one respawn
        # per cooldown window.
        respawn_cooldown = 0

        while step < max_steps:
            if unwrapped.reset_goal_buf[0].item():
                if respawn_cooldown == 0:
                    successes += 1
                    print(f"    [ep{ep:03d}] respawn #{successes} at step {step}")
                    respawn_cooldown = hold_frames
                    # leave buf=True; env consumes it on next _pre_physics_step
                else:
                    # cooldown active -- suppress this respawn trigger
                    unwrapped.reset_goal_buf[0] = False

            if respawn_cooldown > 0:
                respawn_cooldown -= 1

            with torch.no_grad():
                action = get_action(obs_raw)

            obs_dict, reward, terminated, timed_out, _ = env.step(action)
            obs_raw = obs_dict["policy"]
            total_reward += reward[0].item()

            record_frame(writers)

            step += 1
            if (terminated | timed_out)[0].item():
                break

        for writer in writers.values():
            writer.close()

        print(f"  ep{ep:03d} | steps={step:4d} | reward={total_reward:.2f} | successes={successes}")

    env.close()
    simulation_app.close()


def _get_frame(env) -> np.ndarray:
    frame = env.render()
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


if __name__ == "__main__":
    main()