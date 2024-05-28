from rejax.algos.ddpg import DDPG, DDPGConfig
from rejax.algos.dqn import DQN, DQNConfig
from rejax.algos.ppo import PPO, PPOConfig
from rejax.algos.sac import SAC, SACConfig
from rejax.algos.td3 import TD3, TD3Config

_algos = {
    "ppo": (PPO, PPOConfig),
    "dqn": (DQN, DQNConfig),
    "sac": (SAC, SACConfig),
    "ddpg": (DDPG, DDPGConfig),
    "td3": (TD3, TD3Config),
}


def get_agent(agent_str):
    """Gets a pair of `(train_fn, config_cls)`. Will be deprecated in the future, exists
    mainly for backwards compatibility."""
    algo_cls, config_cls = _algos[agent_str]
    return algo_cls.train, config_cls


def get_algo(algo):
    """Get a pair of `(algo_cls, config_cls)` for a given algorithm."""
    return _algos[algo]


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
