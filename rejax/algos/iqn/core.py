import warnings
from copy import deepcopy
from typing import Any, Callable, Type

import chex
import distrax
import gymnax
import jax
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from optax import linear_schedule

from rejax.algos.base_config import BaseConfig
from rejax.algos.iqn.iqn import IQN
from rejax.algos.mixins import EpsilonGreedyMixin, ReplayBufferMixin
from rejax.algos.networks import ImplicitQuantileNetwork
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


def EpsilonGreedyPolicy(iqn: nn.Module) -> Type[nn.Module]:
    class EpsilonGreedyPolicy(iqn):
        def _action_dist(self, obs, rng, epsilon):
            q = self.q(obs, rng)
            return distrax.EpsilonGreedy(q, epsilon=epsilon)

        def act(self, obs, rng, epsilon):
            rng_tau, rng_epsilon = jax.random.split(rng)
            action_dist = self._action_dist(obs, rng_tau, epsilon)
            action = action_dist.sample(seed=rng_epsilon)
            return action

    return EpsilonGreedyPolicy


class IQNConfig(EpsilonGreedyMixin, ReplayBufferMixin, BaseConfig):
    # fmt: off
    # Non-static parameters
    agent: nn.Module                = struct.field(pytree_node=False, default=None)
    target_update_freq: int         = struct.field(pytree_node=True, default=200)
    kappa: chex.Scalar              = struct.field(pytree_node=True, default=1.0)

    # Static parameters
    num_envs: int                   = struct.field(pytree_node=False, default=1)
    gradient_steps: int             = struct.field(pytree_node=False, default=1)
    num_tau_samples: int            = struct.field(pytree_node=False, default=64)
    num_tau_prime_samples: int      = struct.field(pytree_node=False, default=64)
    normalize_observations: bool    = struct.field(pytree_node=False, default=False)
    # fmt: on

    @staticmethod
    def initialize_agent(config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        action_dim = env.action_space(env_params).n
        agent = EpsilonGreedyPolicy(ImplicitQuantileNetwork)(
            hidden_layer_sizes=(64, 64), action_dim=action_dim, **agent_kwargs
        )

        return {"agent": agent}

    @staticmethod
    def initialize_eval_callback(config, env, env_params):
        return make_evaluate(IQN.make_act, env, env_params, 200)
