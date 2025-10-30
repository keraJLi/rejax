from functools import partial
from typing import Any, Union

import chex
import jax
import navix as nx
from gymnax.environments import environment
from jax import numpy as jnp
from navix.environments import wrappers


def create_navix(env_name, **kwargs):
    env = nx.make(env_name, **kwargs)
    env = Navix2GymnaxEnv(env)
    return env, env.default_params


def Navix2GymnaxEnv(env: nx.Environment) -> environment.Environment:  # noqa:  N802
    env = wrappers.ToGymnax(env)
    env = FloatObsWrapper(env)
    return env


class FloatObsWrapper(environment.Environment):
    def __init__(self, env):
        self.env = env

    def __getattribute__(self, name: str) -> Any:
        if name in ["env", "reset", "step", "observation_space"]:
            return super().__getattribute__(name)
        return self.env.__getattribute__(name)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = obs.astype(float)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: environment.EnvParams | None = None
    ) -> tuple[chex.Array, environment.EnvState]:
        obs, state = self.env.reset(key, params)
        obs = obs.astype(float)
        return obs, state

    def observation_space(self, params):
        space = self.env.observation_space(params)
        space.dtype = jnp.float32
        return space
