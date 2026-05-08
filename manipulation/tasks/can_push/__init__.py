import gymnasium as gym

gym.register(
    id="Can-Push",
    entry_point="manipulation.tasks.can_push.env:CanPushEnv",
    disable_env_checker=True,
    kwargs={},
)