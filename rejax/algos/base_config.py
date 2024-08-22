import warnings
from copy import deepcopy
from typing import Any, Callable

import chex
import gymnax
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.brax2gymnax import Brax2GymnaxEnv


class BaseConfig(struct.PyTreeNode):
    env: Environment = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    eval_callback: Callable = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False, default=10_000)
    skip_initial_evaluation: bool = struct.field(pytree_node=False, default=False)

    # Common parameters (excluding algorithm-specific ones)
    total_timesteps: int = struct.field(pytree_node=False, default=100000)
    learning_rate: chex.Scalar = struct.field(pytree_node=True, default=0.005)
    gamma: chex.Scalar = struct.field(pytree_node=True, default=0.99)
    max_grad_norm: chex.Scalar = struct.field(pytree_node=True, default=jnp.inf)

    @classmethod
    def create(cls, **kwargs):
        return cls._from_dict(kwargs)

    @classmethod
    def from_dict(cls, config):
        warnings.warn("from_dict is deprecated, use create instead.")
        return cls._from_dict(config)

    @classmethod
    def _from_dict(cls, config):
        config = deepcopy(config)
        env, env_params = cls.initialize_env(config)
        agent = cls.initialize_agent(config, env, env_params)
        eval_callback = cls.initialize_eval_callback(config, env, env_params)

        return cls(
            env=env,
            env_params=env_params,
            eval_callback=eval_callback,
            **agent,
            **config,
        )

    @staticmethod
    def initialize_env(config):
        if isinstance(config["env"], str):
            env_kwargs = config.pop("env_kwargs", {})
            env_id = config.pop("env")
            if env_id.startswith("brax"):
                env = Brax2GymnaxEnv(env_id.split("/")[1], **env_kwargs)
                env_params = env.default_params
            else:
                env, env_params = gymnax.make(env_id, **env_kwargs)
        else:
            env = config.pop("env")
            env_params = config.pop("env_params", env.default_params)
        return env, env_params

    @staticmethod
    def initialize_agent(config, env, env_params):
        raise NotImplementedError

    @staticmethod
    def initialize_eval_callback(config, env, env_params):
        raise NotImplementedError

    @property
    def discrete(self):
        action_space = self.env.action_space(self.env_params)
        return isinstance(action_space, gymnax.environments.spaces.Discrete)

    @property
    def action_dim(self):
        action_space = self.env.action_space(self.env_params)
        if self.discrete:
            return action_space.n
        return jnp.prod(jnp.array(action_space.shape))
