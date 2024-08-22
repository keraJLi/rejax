import warnings
from copy import deepcopy
from typing import Any, Callable

import chex
import gymnax
import numpy as np
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment

from rejax.algos.base_config import BaseConfig
from rejax.algos.mixins import ReplayBufferMixin
from rejax.algos.networks import (DiscretePolicy, DiscreteQNetwork, QNetwork,
                                  SquashedGaussianPolicy)
from rejax.algos.sac.sac import SAC
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


class SACConfig(ReplayBufferMixin, BaseConfig):
    # fmt: off
    # Non-static parameters
    actor: nn.Module                = struct.field(pytree_node=False, default=None)
    critic: nn.Module               = struct.field(pytree_node=False, default=None)

    tau: float                      = struct.field(pytree_node=True, default=0.95)
    target_entropy_ratio: float     = struct.field(pytree_node=True, default=0.98)
    reward_scaling: chex.Scalar     = struct.field(pytree_node=True, default=1.0)

    # Static parameters
    num_envs: int                   = struct.field(pytree_node=False, default=1)
    gradient_steps: int             = struct.field(pytree_node=False, default=1)
    normalize_observations: bool    = struct.field(pytree_node=False, default=False)
    # fmt: on

    @property
    def target_entropy(self):
        if self.discrete:
            return -self.target_entropy_ratio * np.log(1 / self.action_dim)
        return -self.action_dim

    @property
    def agent(self):
        """Backward compatibility with old evaluation creation"""
        return self.actor

    @staticmethod
    def initialize_agent(config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)
        agent_kwargs = config.pop("agent_kwargs", {})

        # Convert activation from str to Callable
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        # Convert hidden layer sizes to tuple
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
            critic = DiscreteQNetwork(action_dim=action_space.n, **agent_kwargs)
        else:
            actor = SquashedGaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                log_std_range=(-10, 2),
                **agent_kwargs,
            )
            critic = QNetwork(**agent_kwargs)

        return {"actor": actor, "critic": critic}

    @staticmethod
    def initialize_eval_callback(config, env, env_params):
        return make_evaluate(SAC.make_act, env, env_params, 200)
