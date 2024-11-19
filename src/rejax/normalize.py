from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
from flax import struct
from gymnax.environments import environment
from jax import numpy as jnp

"""
Code adapted from
https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
"""


@struct.dataclass
class RMSState:
    mean: chex.Array
    var: chex.Array
    count: int

    @classmethod
    def create(cls, shape):
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=1e-4,
        )


def update_rms(rms_state, obs, batched=True):
    batch = obs if batched else jnp.expand_dims(obs, 0)

    batch_count = batch.shape[0]
    batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)

    delta = batch_mean - rms_state.mean
    tot_count = rms_state.count + batch_count

    new_mean = rms_state.mean + delta * batch_count / tot_count
    m_a = rms_state.var * rms_state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * rms_state.count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return RMSState(mean=new_mean, var=new_var, count=new_count)


def normalize_obs(rms_state, obs):
    return (obs - rms_state.mean) / jnp.sqrt(rms_state.var + 1e-8)


def update_and_normalize(rms_state, obs, batched=True):
    rms_state = update_rms(rms_state, obs, batched)
    return rms_state, normalize_obs(rms_state, obs)


class FloatObsWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name in ["reset", "step"]:
            return super().__getattr__(name)
        return getattr(self.env, name)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = obs.astype(float)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self.env.reset(key, params)
        obs = obs.astype(float)
        return obs, state
