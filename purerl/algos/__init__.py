from purerl.algos.ppo.ppo import train as train_ppo
from purerl.algos.ppo.core import PPOConfig

from purerl.algos.dqn.dqn import train as train_dqn
from purerl.algos.dqn.core import DQNConfig

from purerl.algos.sac.sac import train as train_sac
from purerl.algos.sac.core import SACConfig

from purerl.algos.ddpg.ddpg import train as train_ddpg
from purerl.algos.ddpg.core import DDPGConfig

from purerl.algos.td3.td3 import train as train_td3
from purerl.algos.td3.core import TD3Config


_agents = {
    "ppo": (train_ppo, PPOConfig),
    "dqn": (train_dqn, DQNConfig),
    "sac": (train_sac, SACConfig),
    "ddpg": (train_ddpg, DDPGConfig),
    "td3": (train_td3, TD3Config),
}


def get_agent(agent_str):
    return _agents[agent_str]


__all__ = [
    "get_agent",

    # Algorithms
    "train_ppo",
    "PPOConfig",
    "train_dqn",
    "DQNConfig",
    "train_sac",
    "SACConfig",
    "train_ddpg",
    "DDPGConfig",
    "train_td3",
    "TD3Config",
]
