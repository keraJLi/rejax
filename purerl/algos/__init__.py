from purerl.algos.ppo.ppo import PPO
from purerl.algos.ppo.core import PPOConfig

from purerl.algos.dqn.dqn import DQN
from purerl.algos.dqn.core import DQNConfig

from purerl.algos.sac.sac import SAC
from purerl.algos.sac.core import SACConfig

from purerl.algos.ddpg.ddpg import DDPG
from purerl.algos.ddpg.core import DDPGConfig

from purerl.algos.td3.td3 import TD3
from purerl.algos.td3.core import TD3Config

from purerl.algos.es.es import train as train_es
from purerl.algos.es.core import ESConfig


_agents = {
    "ppo": (PPO.train, PPOConfig),
    "dqn": (DQN.train, DQNConfig),
    "sac": (SAC.train, SACConfig),
    "ddpg": (DDPG.train, DDPGConfig),
    "td3": (TD3.train, TD3Config),
    "es": (train_es, ESConfig),
}


def get_agent(agent_str):
    return _agents[agent_str]


__all__ = [
    "get_agent",

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
    "train_es",
    "ESConfig",
]
