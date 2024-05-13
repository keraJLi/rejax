from purerl.algos.ddpg.core import DDPGConfig
from purerl.algos.ddpg.ddpg import DDPG
from purerl.algos.dqn.core import DQNConfig
from purerl.algos.dqn.dqn import DQN
from purerl.algos.es.core import ESConfig
from purerl.algos.es.es import train as train_es
from purerl.algos.ppo.core import PPOConfig
from purerl.algos.ppo.ppo import PPO
from purerl.algos.sac.core import SACConfig
from purerl.algos.sac.sac import SAC
from purerl.algos.td3.core import TD3Config
from purerl.algos.td3.td3 import TD3

_algos = {
    "ppo": (PPO.train, PPOConfig),
    "dqn": (DQN.train, DQNConfig),
    "sac": (SAC.train, SACConfig),
    "ddpg": (DDPG.train, DDPGConfig),
    "td3": (TD3.train, TD3Config),
    "es": (train_es, ESConfig),
}


def get_algo(agent_str):
    train_fn, config_cls = _algos[agent_str]
    rng_only_train_fn = lambda config, rng: train_fn(config, rng=rng)
    return rng_only_train_fn, config_cls


__all__ = [
    "get_algo",
    # Algorithms
    "PPO",
    "PPOConfig",
    "DQN",
    "DQNConfig",
    "SAC",
    "SACConfig",
    "DDPG",
    "DDPGConfig",
    "TD3",
    "TD3Config",
]
