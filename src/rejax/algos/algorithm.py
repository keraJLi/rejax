from copy import deepcopy
from dataclasses import asdict
from typing import Any, Callable

import chex
import gymnax
import jax
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.compat import create
from rejax.evaluate import evaluate

INIT_REGISTRATION_KEY = "_rejax_registered_init"


def register_init(func):
    setattr(func, INIT_REGISTRATION_KEY, True)
    return func


class Algorithm(struct.PyTreeNode):
    env: Environment = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    eval_callback: Callable = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False, default=4_096)
    skip_initial_evaluation: bool = struct.field(pytree_node=False, default=False)

    # Common parameters (excluding algorithm-specific ones)
    total_timesteps: int = struct.field(pytree_node=False, default=131_072)
    learning_rate: chex.Scalar = struct.field(pytree_node=True, default=0.0003)
    gamma: chex.Scalar = struct.field(pytree_node=True, default=0.99)
    max_grad_norm: chex.Scalar = struct.field(pytree_node=True, default=jnp.inf)

    @classmethod
    def create(cls, **config):
        env, env_params = cls.create_env(config)
        agent = cls.create_agent(config, env, env_params)

        def eval_callback(algo, ts, rng):
            act = algo.make_act(ts)
            max_steps = algo.env_params.max_steps_in_episode
            return evaluate(act, rng, env, env_params, 128, max_steps)

        return cls(
            env=env,
            env_params=env_params,
            eval_callback=eval_callback,
            **agent,
            **config,
        )

    def init_state(self, rng: chex.PRNGKey) -> Any:
        state_values = {}
        for name in dir(self):
            func = getattr(self, name)
            if getattr(func, INIT_REGISTRATION_KEY, False):
                rng, rng_init = jax.random.split(rng, 2)
                state_values.update(func(rng_init))

        cls_name = f"{self.__class__.__name__}State"
        state = {k: struct.field(pytree_node=True) for k in state_values.keys()}
        state_hints = {k: type(v) for k, v in state_values.items()}
        d = {**state, "__annotations__": state_hints}
        clz = type(cls_name, (struct.PyTreeNode,), d)
        return clz(**state_values)

    @register_init
    def init_base_state(self, rng: chex.PRNGKey):
        return {"rng": rng}

    @classmethod
    def create_env(cls, config):
        if isinstance(config["env"], str):
            env, env_params = create(config.pop("env"), **config.pop("env_params", {}))
        else:
            env = config.pop("env")
            env_params = config.pop("env_params", env.default_params)
        return env, env_params

    @classmethod
    def create_agent(cls, config, env, env_params):
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

    @property
    def action_space(self):
        return self.env.action_space(self.env_params)

    @property
    def obs_space(self):
        return self.env.observation_space(self.env_params)

    @property
    def config(self):
        return asdict(self)
