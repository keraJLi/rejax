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
from rejax.algos.mixins import EpsilonGreedyMixin, ReplayBufferMixin
from rejax.algos.networks import DiscreteQNetwork, EpsilonGreedyPolicy
from rejax.algos.pqn.pqn import PQN
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate
from rejax.normalize import FloatObsWrapper


class PQNConfig(EpsilonGreedyMixin, ReplayBufferMixin, BaseConfig):
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    lambda_: chex.Scalar = struct.field(pytree_node=True, default=0.9)
    num_envs: int = struct.field(pytree_node=False, default=16)
    num_steps: int = struct.field(pytree_node=False, default=128)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    num_minibatches: int = struct.field(pytree_node=False, default=8)
    gradient_steps: int = struct.field(pytree_node=False, default=1)
    normalize_observations: bool = struct.field(pytree_node=False, default=False)

    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @staticmethod
    def initialize_agent(config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        agent_kwargs["activation"] = lambda x: nn.relu(nn.LayerNorm()(x))

        action_dim = env.action_space(env_params).n
        agent = EpsilonGreedyPolicy(DiscreteQNetwork)(
            hidden_layer_sizes=(64, 64), action_dim=action_dim, **agent_kwargs
        )
        return {"agent": agent}

    @staticmethod
    def initialize_eval_callback(config, env, env_params):
        return make_evaluate(PQN.make_act, env, env_params, 200)

    @classmethod
    def create(cls, **kwargs):
        config = super().create(**kwargs)
        if config.normalize_observations:
            config = config.replace(env=FloatObsWrapper(config.env))
        return config
