import warnings
from copy import deepcopy
from typing import Any, Callable

import chex
import gymnax
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from optax import linear_schedule

from rejax.algos.base_config import BaseConfig
from rejax.algos.dqn.dqn import DQN
from rejax.algos.mixins import EpsilonGreedyMixin, ReplayBufferMixin
from rejax.algos.networks import (DiscreteQNetwork, DuelingQNetwork,
                                  EpsilonGreedyPolicy)
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


class DQNConfig(EpsilonGreedyMixin, ReplayBufferMixin, BaseConfig):
    # fmt: off
    # Non-static parameters
    agent: nn.Module                = struct.field(pytree_node=False, default=None)
    target_update_freq: int         = struct.field(pytree_node=True, default=200)
    ddqn: bool                      = struct.field(pytree_node=True, default=True)

    # Static parameters
    num_envs: int                   = struct.field(pytree_node=False, default=1)
    gradient_steps: int             = struct.field(pytree_node=False, default=1)
    normalize_observations: bool    = struct.field(pytree_node=False, default=False)
    # fmt: on

    @staticmethod
    def initialize_agent(config, env, env_params):
        agent_name = config.pop("agent", "QNetwork")
        agent_cls = {
            "QNetwork": DiscreteQNetwork,
            "DuelingQNetwork": DuelingQNetwork,
        }[agent_name]
        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        action_dim = env.action_space(env_params).n
        agent = EpsilonGreedyPolicy(agent_cls)(
            hidden_layer_sizes=(64, 64), action_dim=action_dim, **agent_kwargs
        )

        return {"agent": agent}

    @staticmethod
    def initialize_eval_callback(config, env, env_params):
        return make_evaluate(DQN.make_act, env, env_params, 200)
