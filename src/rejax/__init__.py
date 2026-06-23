from rejax.algos import DQN, IQN, MPO, PPO, PQN, SAC, TD3, Algorithm


_algos = {
    "dqn": DQN,
    "iqn": IQN,
    "mpo": MPO,
    "ppo": PPO,
    "pqn": PQN,
    "sac": SAC,
    "td3": TD3,
}


def get_algo(algo: str) -> Algorithm:
    """Get an algorithm class."""
    return _algos[algo]


__all__ = [
    "DQN",
    "IQN",
    "MPO",
    "PPO",
    "PQN",
    "SAC",
    "TD3",
    "get_algo",
]
