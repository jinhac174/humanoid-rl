from algos.ppo.trainer import PPOTrainer
from algos.sapg.trainer import SAPGTrainer
from algos.epo.trainer import EPOTrainer

TRAINER_REGISTRY = {
    "ppo": PPOTrainer,
    "sapg": SAPGTrainer,
    "epo": EPOTrainer,
}