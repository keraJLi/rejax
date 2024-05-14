from purerl.algos.ddpg import DDPG, DDPGConfig
from purerl.algos.dqn import DQN, DQNConfig
from purerl.algos.es import ESConfig, train_es
from purerl.algos.ppo import PPO, PPOConfig
from purerl.algos.sac import SAC, SACConfig
from purerl.algos.td3 import TD3, TD3Config

_algos = {
    "ppo": (PPO.train, PPOConfig),
    "dqn": (DQN.train, DQNConfig),
    "sac": (SAC.train, SACConfig),
    "ddpg": (DDPG.train, DDPGConfig),
    "td3": (TD3.train, TD3Config),
    "es": (train_es, ESConfig),
}


def get_algo(agent_str):
    return _algos[agent_str]


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
