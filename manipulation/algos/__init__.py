from manipulation.algos.ppo.trainer import PPOTrainer
from manipulation.algos.sapg.trainer import SAPGTrainer
from manipulation.algos.epo.trainer import EPOTrainer

TRAINER_REGISTRY = {
    "ppo": PPOTrainer,
    "sapg": SAPGTrainer,
    "epo": EPOTrainer,
}