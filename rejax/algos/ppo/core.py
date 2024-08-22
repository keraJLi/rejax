import warnings
from copy import deepcopy
from typing import Any, Callable

import chex
import gymnax
import numpy as np
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.algos.base_config import BaseConfig
from rejax.algos.networks import DiscretePolicy, GaussianPolicy, VNetwork
from rejax.algos.ppo.ppo import PPO
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


class PPOConfig(BaseConfig):
    # PPO-specific parameters
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)
    num_envs: int = struct.field(pytree_node=False, default=100)
    num_steps: int = struct.field(pytree_node=False, default=50)
    num_epochs: int = struct.field(pytree_node=False, default=5)
    num_minibatches: int = struct.field(pytree_node=False, default=10)
    normalize_observations: bool = struct.field(pytree_node=False, default=False)

    @property
    def minibatch_size(self):
        assert (self.num_envs * self.num_steps) % self.num_minibatches == 0
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @property
    def agent(self):
        """Backward compatibility with old evaluation creation"""
        return self.actor

    @staticmethod
    def initialize_agent(config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
        else:
            actor = GaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                **agent_kwargs,
            )
        critic = VNetwork(**agent_kwargs)

        return {"actor": actor, "critic": critic}

    @staticmethod
    def initialize_eval_callback(config, env, env_params):
        return make_evaluate(PPO.make_act, env, env_params, 200)
