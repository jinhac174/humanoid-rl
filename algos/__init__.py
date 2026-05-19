from algos.ppo.trainer import PPOTrainer
from algos.sapg.trainer import SAPGTrainer
from algos.epo.trainer import EPOTrainer

TRAINER_REGISTRY = {
    "ppo": PPOTrainer,
    "sapg": SAPGTrainer,
    "epo": EPOTrainer,
}

# rsl_rl PPO is an OPTIONAL trainer — it depends on the ``rsl_rl`` Python
# package being installed in the IsaacLab/IsaacSim env (IsaacLab ships it
# as an optional dep). We import lazily so PPO / SAPG / EPO keep working
# on machines that don't have rsl_rl. If a user picks algo=rsl_rl_ppo
# without rsl_rl installed, they get a clear actionable error.
try:
    from algos.rsl_rl_ppo.trainer import RslRlPPOTrainer
    TRAINER_REGISTRY["rsl_rl_ppo"] = RslRlPPOTrainer
except ImportError as _rsl_rl_err:
    _RSL_RL_IMPORT_ERROR = _rsl_rl_err

    def _rsl_rl_unavailable(*args, **kwargs):
        raise ImportError(
            "algo=rsl_rl_ppo requires the 'rsl_rl' package in the IsaacLab "
            "Python env. Install it with:\n"
            "    ~/IsaacLab/isaaclab.sh -p -m pip install rsl-rl-lib\n"
            f"Original import error: {_RSL_RL_IMPORT_ERROR}"
        )

    TRAINER_REGISTRY["rsl_rl_ppo"] = _rsl_rl_unavailable
